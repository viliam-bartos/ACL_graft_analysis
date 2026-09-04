import logging
import numpy as np
import SimpleITK as sitk


def match_histograms(img_sitk, ref_path, mask_sitk=None):
    """
    Match the intensity distribution of the input MRI to the reference MRI
    using Nyul-Udupa histogram standardization via TorchIO.

    Parameters:
        img_sitk (SimpleITK.Image): Input 3D MRI.
        ref_path (str): Path to Reference 3D MRI.
        mask_sitk (SimpleITK.Image, optional): Mask to define non-zero background regions.

    Returns:
        SimpleITK.Image: The standardized MRI.
    """
    logging.info("Starting Module 1: Histogram Matching (Nyul-Udupa via TorchIO).")

    # Lazy import heavy frameworks to keep startup instantaneous
    import torch
    import torchio as tio

    img_array = sitk.GetArrayFromImage(img_sitk)

    # Convert arrays to TorchIO ScalarImages, adding a channel dimension (C, D, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
    img_tio = tio.ScalarImage(tensor=img_tensor)

    # Train landmarks on the reference image path directly
    landmarks = tio.HistogramStandardization.train(
        [str(ref_path)],
        masking_function=lambda x: x > 0
    )

    transform = tio.HistogramStandardization({'mri': landmarks})
    subject = tio.Subject(mri=img_tio)

    if mask_sitk is not None:
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
        subject.add_image(tio.LabelMap(tensor=mask_tensor), 'mask')

    standardized_subject = transform(subject)

    standardized_tensor = standardized_subject['mri'].data.squeeze(0).numpy()
    standardized_sitk = sitk.GetImageFromArray(standardized_tensor)
    standardized_sitk.CopyInformation(img_sitk)

    return standardized_sitk


def extract_radiomics(standardized_img_sitk, mask_sitk):
    """
    Extract first-order, GLCM, and GLRLM radiomics features from the ACL (label 1).

    Parameters:
        standardized_img_sitk (SimpleITK.Image): Standardized input MRI.
        mask_sitk (SimpleITK.Image): Multi-label segmentation mask.

    Returns:
        dict: Extracted radiomics features without diagnostics metadata.
    """
    logging.info("Starting Module 5: Radiomics Extraction.")

    mask_array = sitk.GetArrayFromImage(mask_sitk)
    acl_only_array = (mask_array == 1).astype(np.uint8)
    acl_only_sitk = sitk.GetImageFromArray(acl_only_array)
    acl_only_sitk.CopyInformation(mask_sitk)

    if np.sum(acl_only_array) == 0:
        logging.warning("ACL mask is empty. Skipping radiomics extraction.")
        return {}

    # Lazy import PyRadiomics
    try:
        from radiomics import featureextractor
        logging.getLogger("radiomics").setLevel(logging.ERROR)
    except ImportError:
        logging.error("pyradiomics is not installed. Skipping radiomics extraction.")
        return {}

    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': sitk.sitkBSpline,
        'geometryTolerance': 1e-4,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')

    try:
        features = extractor.execute(standardized_img_sitk, acl_only_sitk)
        extracted = {k: float(v) for k, v in features.items() if not k.startswith('diagnostics')}
        return extracted
    except Exception as e:
        logging.error(f"PyRadiomics extraction failed: {e}")
        return {}
