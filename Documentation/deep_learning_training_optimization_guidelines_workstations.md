# Recommendations – training optimization (CEITEC CTLAB workstations)

Goal: remove the I/O bottleneck (disk) and tune `batch_size` to the maximum that still runs without OOM (VRAM limit).

## Hardware (PC-APOLLO, PC-ATHENA)
- CPU: Intel(R) Xeon(R) w9-3575X (base ~2.21 GHz)
- RAM: 1 TB
- GPU: NVIDIA RTX 6000 Blackwell
- VRAM: 96 GB (verify via `nvidia-smi`)

Monitoring:
- VRAM: `nvidia-smi`
- GPU usage + RAM: Windows Task Manager

## Dataset – storage
- Store as HDF5 (`.h5`).
- Set chunk size to approximately ~4 GB (max.).
- Choose compression based on space vs. speed; for fastest loading prefer no compression (or very light compression).

## Dataset – loading / input pipeline
- If the dataset is < ~900 GB, it can be beneficial to load it entirely into RAM.
- Do not fill RAM to the limit; keep headroom for OS/framework to avoid swapping.
- If data is already fully in RAM as a single contiguous array, set `num_workers = 0` (parallelism often adds overhead in this case).

## Training – tuning `batch_size`
1. Start with `batch_size = 2048`.
2. Increase `batch_size` gradually.
3. When OOM happens, step back by 1 increment.

## After tuning – stability
- Larger batches often improve GPU utilization, but do not always improve convergence.
- After changing `batch_size`, adjust learning rate (rule of thumb):
  $$lr_{new} \approx lr_{base} \cdot \frac{batch_{new}}{batch_{base}}$$
- Monitor VRAM and training stability (e.g., loss divergence).

### Hyperparameter optimization and model size
- For systematic tuning (e.g., `lr`, `batch_size`, weight decay, scheduler, augmentations): Optuna
  - https://optuna.org/
  - https://optuna.readthedocs.io/
- Memory and performance are strongly affected by model size (depth/width); larger models = more VRAM + longer training (not always better quality).
