import os
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import itertools
import csv
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld, RandAffined, Rand3DElasticd, RandGaussianNoised,
    RandAdjustContrastd, RandBiasFieldd, NormalizeIntensityd, SpatialPadd
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.utils.misc import set_determinism
from torch.cuda.amp import autocast, GradScaler

TEST_MODE = True


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)
        self.skip = nn.Identity() if in_c == out_c else nn.Conv3d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))) + self.skip(x))


class LightUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base * 2)
        self.enc3 = ResBlock(base * 2, base * 4)
        self.bottleneck = ResBlock(base * 4, base * 8)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce3 = nn.Conv3d(base * 8, base * 4, kernel_size=1, bias=False)
        self.dec3 = ResBlock(base * 8, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce2 = nn.Conv3d(base * 4, base * 2, kernel_size=1, bias=False)
        self.dec2 = ResBlock(base * 4, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce1 = nn.Conv3d(base * 2, base, kernel_size=1, bias=False)
        self.dec1 = ResBlock(base * 2, base)

        self.final = nn.Conv3d(base, out_ch, kernel_size=1)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bn = self.dropout(self.bottleneck(self.pool(x3)))

        d3 = self.dec3(torch.cat([self.reduce3(self.up3(bn)), x3], dim=1))
        d2 = self.dec2(torch.cat([self.reduce2(self.up2(d3)), x2], dim=1))
        d1 = self.dec1(torch.cat([self.reduce1(self.up1(d2)), x1], dim=1))

        return self.final(d1)


def get_transforms(mode, patch_size):
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    ]

    if mode == 'train':
        augmentations = [
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=patch_size, pos=2, neg=1, num_samples=4
            ),
            RandAffined(
                keys=["image", "label"], prob=0.5,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                mode=("bilinear", "nearest"), padding_mode="zeros"
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 1.5)),
            RandBiasFieldd(keys=["image"], prob=0.2, degree=3, coeff_range=(0.3, 0.5)),
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(5, 8), magnitude_range=(80, 100),
                prob=0.1, mode=("bilinear", "nearest"), padding_mode="zeros"
            ),
        ]
        return Compose(base_transforms + augmentations)
    return Compose(base_transforms)


def save_val_slice(image, gt, pred, save_path):
    img_np = image[0, 0].cpu().numpy()
    gt_np = gt[0, 0].cpu().numpy()
    pred_np = pred[0, 0].cpu().numpy()

    z_mid = img_np.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np[:, :, z_mid], cmap="gray")
    axes[0].set_title("Surové MRI")
    axes[0].axis("off")

    axes[1].imshow(img_np[:, :, z_mid], cmap="gray")
    axes[1].imshow(gt_np[:, :, z_mid], cmap="jet", alpha=0.5)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(img_np[:, :, z_mid], cmap="gray")
    axes[2].imshow(pred_np[:, :, z_mid], cmap="jet", alpha=0.5)
    axes[2].set_title("Predikce sítě")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()


def plot_from_csv(csv_path, save_dir, fold_idx):
    epochs, train_losses, val_losses, dice_scores = [], [], [], []
    iou_scores, hd95_scores, precisions, recalls = [], [], [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_losses.append(float(row['Train_Loss']))
            if row['Val_Loss']: val_losses.append(float(row['Val_Loss']))
            if row['Val_Dice']: dice_scores.append(float(row['Val_Dice']))
            if row['Val_IoU']: iou_scores.append(float(row['Val_IoU']))
            if row['Val_HD95']: hd95_scores.append(float(row['Val_HD95']))
            if row['Val_Precision']: precisions.append(float(row['Val_Precision']))
            if row['Val_Recall']: recalls.append(float(row['Val_Recall']))

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    axs[0].plot(epochs, train_losses, label='Train Loss', color='blue')
    if val_losses:
        val_epochs = [e for e, v in zip(epochs, val_losses) if v]
        axs[0].plot(val_epochs, val_losses, label='Val Loss', color='red', marker='o')
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    if dice_scores:
        axs[1].plot(val_epochs, dice_scores, label='Val Dice', color='green', marker='s')
        axs[1].plot(val_epochs, iou_scores, label='Val IoU', color='orange', marker='^')
        axs[1].set_title('Segmentation Metrics')
        axs[1].set_ylim(0, 1)
        axs[1].legend()
        axs[1].grid(True)

    if precisions:
        axs[2].plot(val_epochs, precisions, label='Precision', color='cyan', marker='v')
        axs[2].plot(val_epochs, recalls, label='Recall', color='magenta', marker='<')
        axs[2].set_title('Precision / Recall')
        axs[2].set_ylim(0, 1)
        axs[2].legend()
        axs[2].grid(True)

    if hd95_scores:
        axs[3].plot(val_epochs, hd95_scores, label='Val HD95', color='purple', marker='d')
        axs[3].set_title('Hausdorff Distance (95%)')
        axs[3].set_ylabel('Distance')
        axs[3].legend()
        axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_fold_{fold_idx}.pdf'), format="pdf")
    plt.close()


def train_fold(config, train_files, val_files, fold_idx, run_dir):
    print(f"\n--- Začíná Fold {fold_idx} ---")

    workers_train = 0 if TEST_MODE else 16
    workers_val = 0 if TEST_MODE else 4

    train_ds = CacheDataset(data=train_files, transform=get_transforms('train', config['patch_size']),
                            cache_rate=1.0, num_workers=workers_val)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=workers_train, pin_memory=not TEST_MODE)

    val_ds = CacheDataset(data=val_files, transform=get_transforms('val', config['patch_size']),
                          cache_rate=1.0, num_workers=workers_val)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers_val)

    model = LightUNet3D(in_ch=1, out_ch=1, base=config['base_filters'])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to('cuda')

    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=config['lr_patience'], verbose=True
    )
    scaler = GradScaler()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    precision_metric = ConfusionMatrixMetric(include_background=False, metric_name="precision", reduction="mean")
    recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall", reduction="mean")

    best_metric = -1
    patience_counter = 0

    csv_path = os.path.join(run_dir, f"log_fold_{fold_idx}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Epoch', 'Epoch_Time_s', 'Train_Loss', 'Val_Loss', 'Val_Dice', 'Val_IoU', 'Val_Precision', 'Val_Recall',
             'Val_HD95', 'Learning_Rate'])

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"F{fold_idx} Ep {epoch + 1}/{config['epochs']} [Train]")
        for batch in pbar:
            images, labels = batch["image"].to('cuda'), batch["label"].to('cuda')

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = round(time.time() - epoch_start_time, 2)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = "", "", "", "", "", ""

        if (epoch + 1) % config['val_interval'] == 0:
            model.eval()
            val_loss_sum = 0
            saved_slice = False

            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to('cuda')
                    val_labels = val_batch["label"].to('cuda')

                    with autocast():
                        val_outputs = sliding_window_inference(
                            inputs=val_images, roi_size=config['patch_size'], sw_batch_size=4,
                            predictor=model, overlap=0.5
                        )
                        v_loss = loss_function(val_outputs, val_labels)
                        val_loss_sum += v_loss.item()

                    val_outputs_bin = (torch.sigmoid(val_outputs) > 0.5).float()

                    dice_metric(y_pred=val_outputs_bin, y=val_labels)
                    iou_metric(y_pred=val_outputs_bin, y=val_labels)
                    precision_metric(y_pred=val_outputs_bin, y=val_labels)
                    recall_metric(y_pred=val_outputs_bin, y=val_labels)
                    hd95_metric(y_pred=val_outputs_bin, y=val_labels)

                    if not saved_slice:
                        pdf_path = os.path.join(run_dir, f"slice_F{fold_idx}_Ep{epoch + 1}.pdf")
                        save_val_slice(val_images, val_labels, val_outputs_bin, pdf_path)
                        saved_slice = True

            val_loss = val_loss_sum / len(val_loader)
            val_dice = dice_metric.aggregate().item()
            val_iou = iou_metric.aggregate().item()

            try:
                val_prec = precision_metric.aggregate()[0].item()
            except:
                val_prec = 0.0

            try:
                val_rec = recall_metric.aggregate()[0].item()
            except:
                val_rec = 0.0

            try:
                val_hd95 = hd95_metric.aggregate().item()
            except:
                val_hd95 = float('inf')

            dice_metric.reset()
            iou_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            hd95_metric.reset()

            print(
                f"Ep {epoch + 1} | Time: {epoch_time}s | T_Loss: {avg_train_loss:.4f} | Dice: {val_dice:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
            scheduler.step(val_dice)

            if val_dice > best_metric:
                best_metric = val_dice
                patience_counter = 0
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, os.path.join(run_dir, f"best_model_fold_{fold_idx}.pth"))
            else:
                patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, epoch_time, avg_train_loss, val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95,
                 current_lr])

        if (epoch + 1) % config['val_interval'] == 0:
            plot_from_csv(csv_path, run_dir, fold_idx)

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_metric': best_metric
        }, os.path.join(run_dir, f"latest_checkpoint_fold_{fold_idx}.pth"))

        if patience_counter >= config['patience']:
            print(f"Early stopping u foldu {fold_idx}")
            break

    return best_metric


def main():
    multiprocessing.freeze_support()
    set_seed(42)

    train_img_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images"
    train_mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels"
    base_save_dir = 'results_grid_cv'
    os.makedirs(base_save_dir, exist_ok=True)

    all_imgs = np.array(sorted(glob.glob(os.path.join(train_img_dir, "*.nii*"))))
    all_masks = np.array(sorted(glob.glob(os.path.join(train_mask_dir, "*.nii*"))))

    if len(all_imgs) == 0:
        raise RuntimeError("Data nenalezena.")

    if TEST_MODE:
        all_imgs = all_imgs[:4]
        all_masks = all_masks[:4]
        param_grid = {
            'patch_size': [(128, 128, 64)],
            'base_filters': [16],
            'lr': [1e-4],
            'epochs': [2],
            'val_interval': [1],
            'batch_size': [2],
            'patience': [2],
            'lr_patience': [1]
        }
    else:
        param_grid = {
            'patch_size': [(128, 128, 64), (96, 96, 96)],
            'base_filters': [32, 64],
            'lr': [1e-4],
            'epochs': [300],
            'val_interval': [5],
            'batch_size': [4],
            'patience': [10],
            'lr_patience': [4]
        }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    kf = KFold(n_splits=5 if not TEST_MODE else 2, shuffle=True, random_state=42)

    for run_idx, config in enumerate(combinations):
        run_name = f"run_{run_idx}_PS{config['patch_size'][2]}_F{config['base_filters']}"
        run_dir = os.path.join(base_save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_imgs)):
            train_files = [{"image": img, "label": mask} for img, mask in
                           zip(all_imgs[train_idx], all_masks[train_idx])]
            val_files = [{"image": img, "label": mask} for img, mask in zip(all_imgs[val_idx], all_masks[val_idx])]

            best_dice = train_fold(config, train_files, val_files, fold_idx, run_dir)
            fold_metrics.append(best_dice)

            if TEST_MODE:
                break

        avg_dice = np.mean(fold_metrics)
        with open(os.path.join(base_save_dir, "grid_search_summary.txt"), "a") as f:
            f.write(f"{run_name} | Avg Dice: {avg_dice:.4f} | Folds: {fold_metrics}\n")


if __name__ == "__main__":
    main()