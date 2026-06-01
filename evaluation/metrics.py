"""
Evaluation Metrics for VintageGAN
Specification Reference: Section 5 - Evaluation Metrics

Implements FID, SSIM, PSNR, and Inception Score for evaluating generated images.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_fid(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    batch_size: int = 50,
    device: str = "cuda",
) -> float:
    """
    Calculate Fréchet Inception Distance (FID).

    Specification Reference: Section 5.1.1 - FID Score

    FID measures the similarity between generated and real image distributions
    using Inception v3 features.

    Args:
        real_images: Real images (N, H, W, 3) uint8 in [0, 255]
        generated_images: Generated images (M, H, W, 3) uint8 in [0, 255]
        batch_size: Batch size for feature extraction
        device: Device to use ('cuda' or 'cpu')

    Returns:
        FID score (lower is better)

    Report only when a licensed real vintage reference set is available.
    """
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        print(
            "Warning: pytorch-fid not installed. Install with: pip install pytorch-fid"
        )
        return -1.0

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize Inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    def get_activations(images, model, batch_size=50, device="cuda"):
        """Extract features from images using Inception v3."""
        model.eval()

        n_images = len(images)
        n_batches = (n_images + batch_size - 1) // batch_size

        pred_arr = np.empty((n_images, 2048))

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_images)

            batch = images[start:end]

            # Convert to tensor and normalize
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0

            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch)[0]

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start:end] = pred

        return pred_arr

    # Extract features
    print("Extracting features from real images...")
    real_features = get_activations(real_images, model, batch_size, device)

    print("Extracting features from generated images...")
    gen_features = get_activations(generated_images, model, batch_size, device)

    # Compute statistics
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(gen_features, axis=0)
    sigma2 = np.cov(gen_features, rowvar=False)

    # Compute FID (spec: Section 5.1.1)
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)


def calculate_ssim(
    generated: np.ndarray, target: np.ndarray, data_range: int = 255
) -> float:
    """
    Calculate Structural Similarity Index (SSIM).

    Specification Reference: Section 5.1.2 - SSIM

    Measures perceptual similarity between images.

    Args:
        generated: Generated images (N, H, W, 3) uint8
        target: Target images (N, H, W, 3) uint8
        data_range: Data range (255 for uint8)

    Returns:
        Average SSIM score (0-1, higher is better)

    Interpret against synthetic targets with caveats; no fixed target is claimed.
    """
    ssim_scores = []

    for gen_img, tgt_img in zip(generated, target):
        # Convert to grayscale for SSIM (spec)
        gen_gray = np.dot(gen_img[..., :3], [0.2989, 0.5870, 0.1140])
        tgt_gray = np.dot(tgt_img[..., :3], [0.2989, 0.5870, 0.1140])

        # Compute SSIM
        score = ssim(tgt_gray, gen_gray, data_range=data_range)
        ssim_scores.append(score)

    return np.mean(ssim_scores)


def calculate_psnr(
    generated: np.ndarray, target: np.ndarray, data_range: int = 255
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Specification Reference: Section 5.1.3 - PSNR

    Measures pixel-level reconstruction quality.

    Args:
        generated: Generated images (N, H, W, 3) uint8
        target: Target images (N, H, W, 3) uint8
        data_range: Data range (255 for uint8)

    Returns:
        Average PSNR in dB (higher is better)

    Interpret against synthetic targets with caveats; no fixed target is claimed.
    """
    psnr_scores = []

    for gen_img, tgt_img in zip(generated, target):
        score = psnr(tgt_img, gen_img, data_range=data_range)
        psnr_scores.append(score)

    return np.mean(psnr_scores)


def calculate_inception_score(
    images: np.ndarray, batch_size: int = 32, splits: int = 10, device: str = "cuda"
) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS).

    Specification Reference: Section 5.1.4 - Inception Score

    Measures quality and diversity of generated images.

    Args:
        images: Generated images (N, H, W, 3) uint8
        batch_size: Batch size
        splits: Number of splits for computing std
        device: Device to use

    Returns:
        Tuple of (mean_score, std_score)

    Target: IS > 3.0 (Specification Section 5.1.4)
    """
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        print("Warning: pytorch-fid not installed")
        return -1.0, -1.0

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize Inception model
    inception_model = InceptionV3([3]).to(device)
    inception_model.eval()

    n_images = len(images)
    preds = []

    for i in range(0, n_images, batch_size):
        batch = images[i : i + batch_size]

        # Convert to tensor
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0

        batch = batch.to(device)

        with torch.no_grad():
            pred = inception_model(batch)[0]

        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)

    # Calculate IS
    split_scores = []

    for k in range(splits):
        part = preds[k * (n_images // splits) : (k + 1) * (n_images // splits)]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_condition_accuracy(
    generator: nn.Module,
    detectors: dict,
    test_images: torch.Tensor,
    test_conditions: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Calculate condition accuracy (MAE).

    Specification Reference: Section 5.1.5 - Condition Accuracy

    Verifies that generated images match requested conditions.

    Args:
        generator: Trained generator
        detectors: Defect detector networks
        test_images: Test images (N, 3, H, W) in [-1, 1]
        test_conditions: Target conditions (N, 6) in [0, 1]
        device: Device to use

    Returns:
        Mean Absolute Error (lower is better)

    Target: MAE < 0.15 (Specification Section 5.1.5)
    """
    generator.eval()

    # Move to device
    test_images = test_images.to(device)
    test_conditions = test_conditions.to(device)

    # Generate images
    with torch.no_grad():
        generated = generator(test_images, test_conditions)

    # Extract conditions using detectors
    if "multi" in detectors:
        # Multi-task detector
        detector = detectors["multi"].to(device)
        detector.eval()

        with torch.no_grad():
            detected = detector(generated)
    else:
        # Separate detectors
        detected_list = []
        for defect_name in [
            "grain",
            "scratch",
            "dust",
            "vignette",
            "color_shift",
            "blur",
        ]:
            if defect_name in detectors:
                detector = detectors[defect_name].to(device)
                detector.eval()

                with torch.no_grad():
                    pred = detector(generated)
                detected_list.append(pred)
            else:
                # No detector available
                detected_list.append(
                    test_conditions[:, len(detected_list) : len(detected_list) + 1]
                )

        detected = torch.cat(detected_list, dim=1)

    # Calculate MAE
    mae = torch.abs(detected - test_conditions).mean().item()

    return mae


def evaluate_model(
    generator: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    real_images_path: str = None,
    detectors: dict = None,
    device: str = "cuda",
) -> dict:
    """
    Comprehensive model evaluation.

    Computes all evaluation metrics.

    Args:
        generator: Trained generator
        test_loader: Test dataloader
        real_images_path: Path to real vintage images for FID
        detectors: Detector networks for condition accuracy
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    generator.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    # Collect generated and target images
    generated_images = []
    target_images = []
    clean_images = []
    all_conditions = []

    print("\nGenerating test images...")
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                clean = batch["clean"].to(device)
                defected = batch["defected"].to(device)
                conditions = batch["condition"].to(device)
            else:
                images, _ = batch
                clean = images.to(device)
                defected = images
                conditions = torch.rand(clean.size(0), 6)
                conditions = conditions.to(device)

            # Generate
            generated = generator(clean, conditions)

            # Convert to numpy uint8
            gen_np = ((generated + 1) / 2 * 255).clamp(0, 255).byte().cpu()
            gen_np = gen_np.permute(0, 2, 3, 1).numpy()

            tgt_np = ((defected + 1) / 2 * 255).clamp(0, 255).byte().cpu()
            tgt_np = tgt_np.permute(0, 2, 3, 1).numpy()

            clean_np = ((clean + 1) / 2 * 255).clamp(0, 255).byte().cpu()
            clean_np = clean_np.permute(0, 2, 3, 1).numpy()

            generated_images.append(gen_np)
            target_images.append(tgt_np)
            clean_images.append(clean_np)
            all_conditions.append(conditions.cpu())

    generated_images = np.concatenate(generated_images, axis=0)
    target_images = np.concatenate(target_images, axis=0)
    clean_images = np.concatenate(clean_images, axis=0)
    all_conditions = torch.cat(all_conditions, dim=0)

    print(f"Generated {len(generated_images)} test images")

    # Calculate metrics
    metrics = {}

    # SSIM
    print("\nCalculating SSIM...")
    ssim_score = calculate_ssim(generated_images, target_images)
    metrics["ssim"] = ssim_score
    print(f"SSIM: {ssim_score:.4f}")

    # PSNR
    print("\nCalculating PSNR...")
    psnr_score = calculate_psnr(generated_images, target_images)
    metrics["psnr"] = psnr_score
    print(f"PSNR: {psnr_score:.2f} dB")

    # FID (if real images provided)
    if real_images_path is not None:
        print("\nCalculating FID...")
        # Load real images
        from pathlib import Path

        real_imgs = []
        target_size = generated_images.shape[1]
        for img_path in Path(real_images_path).glob("*.jpg"):
            img = np.array(Image.open(img_path).resize((target_size, target_size)))
            real_imgs.append(img)

        if len(real_imgs) > 0:
            real_imgs = np.stack(real_imgs[: len(generated_images)])
            fid_score = calculate_fid(real_imgs, generated_images, device=device)
            metrics["fid"] = fid_score
            print(f"FID: {fid_score:.2f}")
        else:
            print("No real images found, skipping FID")

    # Condition accuracy (if detectors provided)
    if detectors is not None:
        print("\nCalculating condition accuracy...")
        # Sample subset for efficiency
        n_samples = min(100, len(generated_images))
        idx = np.random.choice(len(generated_images), n_samples, replace=False)

        sample_clean = (
            torch.from_numpy(clean_images[idx]).permute(0, 3, 1, 2).float() / 127.5 - 1
        )
        sample_cond = all_conditions[idx]

        mae = calculate_condition_accuracy(
            generator, detectors, sample_clean, sample_cond, device
        )
        metrics["condition_mae"] = mae
        print(f"Condition MAE: {mae:.4f} (target: <0.15)")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    """Test script for evaluation metrics."""
    print("Evaluation Metrics - Test Script")
    print("=" * 60)

    # Create dummy images
    print("\nGenerating dummy images for testing...")
    real_imgs = np.random.randint(0, 256, (100, 256, 256, 3), dtype=np.uint8)
    gen_imgs = np.random.randint(0, 256, (100, 256, 256, 3), dtype=np.uint8)

    # Test SSIM
    print("\n1. Testing SSIM...")
    try:
        ssim_score = calculate_ssim(gen_imgs[:10], real_imgs[:10])
        print(f"   ✓ SSIM: {ssim_score:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test PSNR
    print("\n2. Testing PSNR...")
    try:
        psnr_score = calculate_psnr(gen_imgs[:10], real_imgs[:10])
        print(f"   ✓ PSNR: {psnr_score:.2f} dB")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n✅ Evaluation metrics tests complete!")
    print("=" * 60)
