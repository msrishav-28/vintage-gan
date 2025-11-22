# COMPLETE SPECIFICATION DOCUMENT: CONTROLLABLE VINTAGE FILM DEFECT SYNTHESIS

**Project Name:** VintageGAN - Parametric Film Defect Generator  
**Model Type:** Conditional GAN with Multi-Parameter Control  
**Framework:** PyTorch 2.0+  
**Target Hardware:** NVIDIA RTX 3050 (6GB VRAM), 16GB RAM  
**Development Tool:** Claude Sonnet 4.5 via Windsurf IDE  
**Timeline:** 12-15 days for academic project completion

***

## 1. PROJECT OBJECTIVES

### 1.1 Primary Goal
Build a conditional Generative Adversarial Network that applies controllable vintage film defects to digital images, where users can independently adjust intensity parameters for each defect type (grain, scratches, dust, vignetting, color shift, blur).[1][2][3]

### 1.2 Academic Contribution
Extend existing Film-GAN approaches  with **parametric control** via conditioning vectors, filling the gap between fixed-preset filters and fully controllable synthesis.[4][2][3][5]

### 1.3 Success Criteria
- **Quantitative:** FID < 50, SSIM > 0.75, PSNR > 22 dB on validation set[6][7]
- **Qualitative:** Independent control over 6 defect parameters with smooth intensity gradients[2][3]
- **Performance:** Inference < 2 seconds per 512×512 image on RTX 3050[8]
- **Academic:** Complete documentation meeting B.Tech standards[9][10]

---

## 2. ARCHITECTURE SPECIFICATION

### 2.1 Generator Architecture

**Base Model:** U-Net with ResNet34 encoder[11][1]

**Architecture Details:**
```
Input: RGB Image (512×512×3) + Conditioning Vector (6D)
Encoder Path (ResNet34 backbone):
  - Conv Block 1: 64 filters, stride 2 → (256×256×64)
  - Conv Block 2: 128 filters, stride 2 → (128×128×128)
  - Conv Block 3: 256 filters, stride 2 → (64×64×256)
  - Conv Block 4: 512 filters, stride 2 → (32×32×512)
  
Bottleneck:
  - Self-Attention Layer (channel attention) [web:15]
  - Condition Integration Module (see 2.1.1)
  
Decoder Path:
  - UpConv Block 4: 512→256 filters → (64×64×256)
  - UpConv Block 3: 256→128 filters → (128×128×128)
  - UpConv Block 2: 128→64 filters → (256×256×64)
  - UpConv Block 1: 64→3 filters → (512×512×3)
  
Skip Connections: Concatenate encoder features to decoder at each level
Output Activation: Tanh (normalize to [-1, 1])
```

**Implementation Notes:**
- Use spectral normalization on all convolutional layers to stabilize training[12][11]
- Apply dropout (p=0.3) in decoder blocks to prevent overfitting[11]
- Initialize weights with He initialization for ReLU activations[11]

### 2.1.1 Conditioning Vector Integration[13][1][4]

**Conditioning Vector Format:**
```python
condition_vector = [
    grain_intensity,      # [0.0, 1.0] - Film grain amount
    scratch_density,      # [0.0, 1.0] - Scratch frequency
    dust_count,           # [0.0, 1.0] - Dust particle density
    vignette_strength,    # [0.0, 1.0] - Vignetting darkness
    color_shift,          # [0.0, 1.0] - Color degradation level
    blur_amount           # [0.0, 1.0] - Motion/defocus blur
]
```

**Integration Method (Concatenation-Based Conditioning):**[4][13]

1. **Input Concatenation:**
   - Spatially replicate 6D vector to (512×512×6) tensor[4]
   - Concatenate with RGB input → (512×512×9)[4]
   - First convolution layer processes 9 channels instead of 3[4]

2. **Intermediate Feature Modulation:**
   - Project condition vector through MLP: 6 → 128 → 512 dimensions[1][4]
   - At bottleneck: reshape to (32×32×512) and concatenate with encoder output[4]
   - This dual-injection ensures conditions influence both early and late features[1][4]

**Code Structure:**
```python
class ConditionProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )
    
    def forward(self, condition_vec):
        # Project and reshape for concatenation
        embedded = self.mlp(condition_vec)  # (batch, 512)
        return embedded.view(-1, 512, 1, 1).expand(-1, -1, 32, 32)
```

### 2.2 Discriminator Architecture

**Type:** PatchGAN Discriminator with Conditional Input[1][4]

**Architecture Details:**
```
Input: RGB Image (512×512×3) + Condition Vector (6D replicated spatially)
Concatenated Input: (512×512×9)

Conv Layer 1: 64 filters, kernel 4×4, stride 2 → (256×256×64)
Conv Layer 2: 128 filters, kernel 4×4, stride 2 → (128×128×128)
Conv Layer 3: 256 filters, kernel 4×4, stride 2 → (64×64×256)
Conv Layer 4: 512 filters, kernel 4×4, stride 2 → (32×32×512)
Conv Layer 5: 1 filter, kernel 4×4, stride 1 → (32×32×1)

Output: 32×32 patch predictions (real/fake per patch)
```

**Activation Functions:**
- LeakyReLU (α=0.2) for all layers except output[11]
- No activation on final layer (raw logits)[11]

**Implementation Notes:**
- Apply spectral normalization to all conv layers[12]
- Use InstanceNorm instead of BatchNorm for better GAN stability[11]
- Condition vector is spatially replicated and concatenated with image input[1][4]

### 2.3 Self-Attention Module[11]

**Location:** Bottleneck of generator (32×32 feature resolution)

**Implementation:**
```python
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H*W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(B, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x
```

**Rationale:** Captures long-range dependencies for scratch patterns and vignetting[14][11]

---

## 3. DATASET CREATION

### 3.1 Base Dataset

**Source:** ImageNet 1K subset[11]
- **Size:** 10,000 training images, 1,000 validation images[15]
- **Categories:** Diverse natural images (landscapes, portraits, objects, architecture)
- **Resolution:** All images resized to 512×512 with center crop[16]
- **Format:** JPEG/PNG converted to normalized tensors [-1, 1][16]

**Download Instructions:**
```bash
# Use Kaggle API or Hugging Face datasets
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", split="train", streaming=True)
# Sample 10k images randomly with seed=42 for reproducibility
```

### 3.2 Synthetic Defect Generation

For each clean image, generate paired defected versions with ground-truth condition labels.[17][18][16]

#### 3.2.1 Film Grain Synthesis[19][20]

**Algorithm:** Frequency-based grain pattern generation[19]

```python
def generate_film_grain(image, intensity):
    """
    intensity: 0.0-1.0 controlling grain visibility
    """
    h, w = image.shape[:2]
    # Generate Gaussian noise base
    noise = np.random.normal(0, intensity * 25, (h, w))
    
    # Apply frequency filtering (film grain has characteristic spectrum)
    freq_limit = 0.3 + (intensity * 0.4)  # Higher intensity = coarser grain
    noise_fft = np.fft.fft2(noise)
    freq_mask = create_bandpass_filter(h, w, low=0.1, high=freq_limit)
    filtered_noise = np.real(np.fft.ifft2(noise_fft * freq_mask))
    
    # Add to image with luma blending
    grain_layer = filtered_noise[:, :, np.newaxis]
    defected = image + grain_layer
    return np.clip(defected, 0, 255).astype(np.uint8)
```

**Parameters:**
- Low intensity (0.0-0.3): Fine grain, ISO 100-400 equivalent
- Medium (0.4-0.6): Noticeable grain, ISO 800-1600
- High (0.7-1.0): Heavy grain, ISO 3200+ or pushed film

#### 3.2.2 Scratch Generation[18][17]

**Algorithm:** Morphological line defects[17][18]

```python
def generate_scratches(image, density):
    """
    density: 0.0-1.0 controlling number of scratches
    """
    h, w = image.shape[:2]
    num_scratches = int(density * 15)  # 0-15 scratches per image
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_scratches):
        # Vertical scratches (most common in film)
        x_pos = np.random.randint(0, w)
        thickness = np.random.randint(1, 4)
        
        # Scratches don't always span full height
        y_start = np.random.randint(0, h // 3)
        y_end = np.random.randint(2 * h // 3, h)
        
        cv2.line(scratch_mask, (x_pos, y_start), (x_pos, y_end), 
                 255, thickness)
    
    # Apply scratches as dark/light lines
    scratch_intensity = np.random.choice([0.3, 0.7])  # Dark or light
    defected = image.copy()
    defected[scratch_mask > 0] = image[scratch_mask > 0] * scratch_intensity
    return defected
```

**Statistical Properties:**[18]
- Width: 1-3 pixels (99% of analog scratches)
- Orientation: 85-90° vertical bias (film movement direction)
- Color: 70% dark scratches, 30% light (emulsion vs base damage)

#### 3.2.3 Dust Particle Synthesis[17][18]

**Algorithm:** Random elliptical particles[18]

```python
def generate_dust(image, count):
    """
    count: 0.0-1.0 controlling particle density
    """
    h, w = image.shape[:2]
    num_particles = int(count * 50)  # 0-50 particles
    dust_mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_particles):
        # Random position, size, orientation
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(2, 8), np.random.randint(2, 8))
        angle = np.random.randint(0, 180)
        
        cv2.ellipse(dust_mask, center, axes, angle, 0, 360, 255, -1)
    
    # Apply as dark spots with soft edges
    dust_mask_blur = cv2.GaussianBlur(dust_mask, (5, 5), 2)
    defected = image.copy()
    alpha = (dust_mask_blur / 255.0)[:, :, np.newaxis] * 0.5
    defected = image * (1 - alpha) + (image * 0.3) * alpha
    return defected.astype(np.uint8)
```

#### 3.2.4 Vignetting[21]

**Algorithm:** Radial gradient darkening

```python
def generate_vignette(image, strength):
    """
    strength: 0.0-1.0 controlling vignette intensity
    """
    h, w = image.shape[:2]
    
    # Create radial gradient from center
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Normalize to [0, 1]
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette_mask = dist_from_center / max_dist
    
    # Apply power function for natural falloff
    vignette_mask = vignette_mask ** (1.5 + strength)
    vignette_mask = 1 - (vignette_mask * strength * 0.7)
    
    # Apply to image
    vignette_mask = vignette_mask[:, :, np.newaxis]
    defected = image * vignette_mask
    return np.clip(defected, 0, 255).astype(np.uint8)
```

#### 3.2.5 Color Shift[22][21]

**Algorithm:** Vintage film stock color degradation

```python
def generate_color_shift(image, shift_amount):
    """
    shift_amount: 0.0-1.0 controlling color degradation
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Shift toward sepia/faded tones
    a = a * (1 - shift_amount * 0.4) + shift_amount * 20
    b = b * (1 - shift_amount * 0.4) + shift_amount * 30
    
    # Reduce saturation
    a = a * (1 - shift_amount * 0.3)
    b = b * (1 - shift_amount * 0.3)
    
    # Merge and convert back
    lab = cv2.merge([l, a.astype(np.uint8), b.astype(np.uint8)])
    defected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return defected
```

#### 3.2.6 Blur[21]

**Algorithm:** Combined motion and lens blur

```python
def generate_blur(image, amount):
    """
    amount: 0.0-1.0 controlling blur strength
    """
    if amount < 0.1:
        return image
    
    kernel_size = int(3 + amount * 10)  # 3-13 pixels
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Random choice between motion blur and Gaussian
    if np.random.random() < 0.5:
        # Motion blur
        angle = np.random.randint(0, 180)
        kernel = create_motion_blur_kernel(kernel_size, angle)
        defected = cv2.filter2D(image, -1, kernel)
    else:
        # Gaussian (lens defocus)
        sigma = amount * 3
        defected = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return defected
```

### 3.3 Training Pair Generation[16]

**Process:** For each clean image, generate multiple variants with random condition combinations

```python
def create_training_pair():
    """
    Returns: (clean_image, defected_image, condition_vector)
    """
    # Load random clean image
    clean_img = load_random_imagenet_sample()
    
    # Sample random condition values
    conditions = {
        'grain': np.random.uniform(0.0, 1.0),
        'scratch': np.random.uniform(0.0, 1.0),
        'dust': np.random.uniform(0.0, 1.0),
        'vignette': np.random.uniform(0.0, 1.0),
        'color_shift': np.random.uniform(0.0, 1.0),
        'blur': np.random.uniform(0.0, 1.0)
    }
    
    # Apply defects in order (order matters!)
    defected = clean_img.copy()
    defected = generate_film_grain(defected, conditions['grain'])
    defected = generate_scratches(defected, conditions['scratch'])
    defected = generate_dust(defected, conditions['dust'])
    defected = generate_vignette(defected, conditions['vignette'])
    defected = generate_color_shift(defected, conditions['color_shift'])
    defected = generate_blur(defected, conditions['blur'])
    
    condition_vec = np.array([
        conditions['grain'],
        conditions['scratch'],
        conditions['dust'],
        conditions['vignette'],
        conditions['color_shift'],
        conditions['blur']
    ], dtype=np.float32)
    
    return clean_img, defected, condition_vec
```

**Dataset Size:**
- Training: 10,000 clean images × 5 defect variants each = 50,000 pairs[22][16]
- Validation: 1,000 clean images × 3 defect variants = 3,000 pairs[16]

**Data Augmentation:**[16]
- Random horizontal flips (p=0.5)
- Random crops (480×480 then resize to 512×512)
- Color jitter (brightness ±10%, contrast ±10%) applied to BOTH clean and defected

***

## 4. TRAINING METHODOLOGY

### 4.1 NoGAN Training Strategy[12][11]

**Phase 1: Generator Pretraining (Epochs 1-40)**

**Loss Function:** Perceptual Loss + Pixel Loss
```python
# Feature loss using VGG16
vgg_features = VGG16(pretrained=True).features[:23]  # Up to conv4_3
vgg_features.eval()
for param in vgg_features.parameters():
    param.requires_grad = False

def perceptual_loss(generated, target):
    gen_features = vgg_features(generated)
    target_features = vgg_features(target)
    return F.mse_loss(gen_features, target_features)

def pixel_loss(generated, target):
    return F.l1_loss(generated, target)

# Combined pretraining loss
loss_pretrain = pixel_loss(gen_output, target) + 0.1 * perceptual_loss(gen_output, target)
```

**Hyperparameters:**
- Optimizer: Adam with β₁=0.5, β₂=0.999[11]
- Learning Rate: 2e-4 with cosine annealing[11]
- Batch Size: 8 (fits in 6GB VRAM with mixed precision)[11]
- Gradient Accumulation: 4 steps (effective batch size 32)[11]
- Mixed Precision: FP16 training with GradScaler[11]

**Phase 2: Critic Pretraining (Epochs 41-50)**

Freeze generator, train discriminator to distinguish real vintage photos (from FilmSet ) vs generator outputs[22]

```python
# Binary cross-entropy for discriminator
real_labels = torch.ones(batch_size, 32, 32)
fake_labels = torch.zeros(batch_size, 32, 32)

d_loss_real = F.binary_cross_entropy_with_logits(
    discriminator(real_vintage_photos, conditions), real_labels)
d_loss_fake = F.binary_cross_entropy_with_logits(
    discriminator(generator(clean_images, conditions), conditions), fake_labels)

d_loss = d_loss_real + d_loss_fake
```

**Phase 3: GAN Fine-Tuning (Epochs 51-60)**[12][11]

**Generator Loss:**
```python
# Adversarial loss
gen_output = generator(clean_images, conditions)
d_pred_fake = discriminator(gen_output, conditions)
adv_loss = F.binary_cross_entropy_with_logits(d_pred_fake, real_labels)

# Consistency loss (ControlNet++ technique)
# Extract conditions from generated images using pre-trained extractors
extracted_grain = grain_detector(gen_output)  # 0-1 score
extracted_vignette = vignette_detector(gen_output)  # 0-1 score
# ... other extractors

consistency_loss = F.mse_loss(
    torch.stack([extracted_grain, extracted_vignette, ...]),
    conditions
)

# Total generator loss
g_loss = adv_loss + 0.5 * pixel_loss(gen_output, target) + \
         0.1 * perceptual_loss(gen_output, target) + \
         0.3 * consistency_loss
```

**Discriminator Loss:**
```python
# Standard GAN discriminator loss with label smoothing
real_labels_smooth = torch.ones(batch_size, 32, 32) * 0.9
fake_labels_smooth = torch.zeros(batch_size, 32, 32) + 0.1

d_loss_real = F.binary_cross_entropy_with_logits(
    discriminator(real_vintage, conditions), real_labels_smooth)
d_loss_fake = F.binary_cross_entropy_with_logits(
    discriminator(gen_output.detach(), conditions), fake_labels_smooth)

d_loss = d_loss_real + d_loss_fake
```

**Training Schedule:**
- Update discriminator 1 time per batch[11]
- Update generator 1 time per batch[11]
- Learning rate: 2e-5 (10× lower than pretraining)[11]

**Early Stopping:** Stop GAN training if FID score plateaus for 3 epochs[11]

### 4.2 Consistency Loss Implementation[23][12]

**Purpose:** Enforce that generated images actually contain requested defect levels[23][12]

**Defect Extractor Networks:** Train lightweight classifiers to detect defect levels

```python
class GrainDetector(nn.Module):
    """Predicts grain intensity from image"""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 1)  # Single regression output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image):
        return self.sigmoid(self.backbone(image))

# Train on synthetic dataset with known grain levels
# Use MSE loss between predicted and ground-truth grain intensity
```

**Train 6 separate extractors** (one per defect type) on synthetic dataset before GAN training[23][12]

**Integration during GAN Training:**
```python
# During generator update
gen_image = generator(clean, conditions)

# Extract defects from generated image
detected_conditions = torch.stack([
    grain_detector(gen_image),
    scratch_detector(gen_image),
    dust_detector(gen_image),
    vignette_detector(gen_image),
    color_shift_detector(gen_image),
    blur_detector(gen_image)
], dim=1).squeeze()  # Shape: (batch, 6)

# Consistency loss
consistency_loss = F.mse_loss(detected_conditions, conditions)
```

**Weight in Total Loss:** λ = 0.3 (30% of combined loss)[23][12]

### 4.3 Training Schedule

**Total Epochs:** 60
- Epochs 1-40: Generator pretraining (5-6 hours on RTX 3050)[11]
- Epochs 41-50: Discriminator pretraining (1-2 hours)[11]
- Epochs 51-60: GAN fine-tuning (2-3 hours)[11]

**Total Training Time:** ~10 hours[11]

**Checkpointing:**
- Save model every 5 epochs[16]
- Keep best model based on validation FID score[7]
- Save optimizer state for resuming training[16]

**Monitoring Metrics:**
- Training Loss (generator and discriminator)
- Validation FID score (every epoch)[7]
- Validation SSIM and PSNR (every epoch)[6]
- Sample images (save 10 validation outputs every 2 epochs)[16]

***

## 5. EVALUATION METRICS

### 5.1 Quantitative Metrics[24][6][7]

#### 5.1.1 Fréchet Inception Distance (FID)[24][7]

**Purpose:** Measures similarity between generated vintage images and real vintage photos[7][24]

**Implementation:**
```python
from pytorch_fid import fid_score

# Generate 3000 images with random conditions
# Compare against FilmSet dataset (real vintage photos)
fid_value = fid_score.calculate_fid_given_paths(
    [generated_images_dir, filmset_dir],
    batch_size=50,
    device='cuda',
    dims=2048
)
```

**Target:** FID < 50 (excellent), < 30 (publication-quality)[7]

**Frequency:** Calculate every epoch on validation set[7]

#### 5.1.2 Structural Similarity Index (SSIM)[6]

**Purpose:** Measures perceptual similarity between input and output[6]

**Implementation:**
```python
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(generated, target):
    # Convert to grayscale for SSIM
    gen_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    
    score = ssim(gen_gray, target_gray, data_range=255)
    return score

# Average across validation set
avg_ssim = np.mean([calculate_ssim(gen, tgt) for gen, tgt in val_pairs])
```

**Target:** SSIM > 0.75[6]

**Range:** 0 (completely different) to 1 (identical)[6]

#### 5.1.3 Peak Signal-to-Noise Ratio (PSNR)[6]

**Purpose:** Measures pixel-level reconstruction quality[6]

**Implementation:**
```python
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(generated, target):
    return psnr(target, generated, data_range=255)

avg_psnr = np.mean([calculate_psnr(gen, tgt) for gen, tgt in val_pairs])
```

**Target:** PSNR > 22 dB[6]

**Range:** Higher is better; 20-25 dB typical for style transfer[6]

#### 5.1.4 Inception Score (IS)[24][7]

**Purpose:** Measures quality and diversity of generated images[24][7]

**Implementation:**
```python
from pytorch_fid.inception import InceptionV3
import torch.nn.functional as F

def calculate_inception_score(images, batch_size=32, splits=10):
    inception_model = InceptionV3([3]).cuda()
    inception_model.eval()
    
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].cuda()
        with torch.no_grad():
            pred = F.softmax(inception_model(batch)[0], dim=1)
        preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

is_mean, is_std = calculate_inception_score(generated_images)
```

**Target:** IS > 3.0[7]

#### 5.1.5 Condition Accuracy[12][23]

**Purpose:** Verify generated images match requested conditions[12]

**Implementation:**
```python
def evaluate_condition_accuracy(generator, test_conditions):
    """
    test_conditions: List of (clean_image, target_conditions) pairs
    """
    errors = []
    
    for clean, target_cond in test_conditions:
        # Generate with target conditions
        gen_image = generator(clean, target_cond)
        
        # Extract actual conditions using detectors
        detected = torch.stack([
            grain_detector(gen_image),
            scratch_detector(gen_image),
            # ... other detectors
        ])
        
        # Calculate MAE
        error = torch.abs(detected - target_cond).mean()
        errors.append(error.item())
    
    return np.mean(errors)

# Target: MAE < 0.15 (conditions accurate within 15%)
```

### 5.2 Ablation Studies[25][26]

**Purpose:** Prove each component contributes to performance[25]

**Variants to Test:**

1. **Baseline:** U-Net without conditioning (single global style)
2. **No Consistency Loss:** Remove consistency loss term
3. **No GAN Training:** Pretraining only (NoGAN extreme)
4. **No Self-Attention:** Remove attention module
5. **Single Condition:** Train with only grain control
6. **Full Model:** Complete architecture with all components

**Evaluation:** Compare all variants using FID, SSIM, PSNR on same validation set[7]

**Expected Result:** Full model achieves best FID/SSIM scores[25]

### 5.3 User Study[27][28]

**Protocol:** Perceptual evaluation with human participants[28]

**Design:**
- **Participants:** 25-30 people (mix of photographers and general users)[28]
- **Task:** Rate 50 image pairs on 5-point Likert scale[28]
  - "How realistic is the vintage effect?" (1=fake, 5=authentic)
- **Comparison:** Show triplets (original, our method, baseline method)[28]
- **Duration:** 15-20 minutes per participant[28]

**Analysis:**
- Calculate mean opinion score (MOS)[28]
- Perform paired t-test comparing our method vs baseline[7]
- Calculate Cohen's Kappa for inter-rater agreement[28]

**Target:** MOS > 4.0, statistically significant improvement over baseline (p < 0.05)[28]

### 5.4 Baseline Comparisons[29][24]

**Compare against:**

1. **Traditional Filters:** Photoshop vintage presets, VSCO filters[24]
2. **CycleGAN:** Unpaired image-to-image translation[24]
3. **Film-GAN:** Existing film simulation model[5]

**Metrics for Comparison:**
- FID, SSIM, PSNR (quantitative)[24][7]
- User preference score (qualitative)[28]
- Inference time (efficiency)[24]

**Present Results in Table:**
```
Method          | FID ↓  | SSIM ↑ | PSNR ↑ | Inference Time
----------------|--------|--------|--------|----------------
VSCO Filter     | 75.3   | 0.68   | 19.2   | 0.01s
CycleGAN        | 52.1   | 0.71   | 21.5   | 1.8s
Film-GAN        | 48.6   | 0.73   | 22.1   | 1.5s
Ours (VintageGAN)| 42.3   | 0.78   | 23.7   | 1.2s
```

***

## 6. IMPLEMENTATION REQUIREMENTS

### 6.1 Software Dependencies

```
# Core Framework
python >= 3.9
pytorch >= 2.0.0
torchvision >= 0.15.0
cuda >= 11.8

# Computer Vision
opencv-python >= 4.8.0
scikit-image >= 0.21.0
pillow >= 10.0.0

# Evaluation
pytorch-fid >= 0.3.0
lpips >= 0.1.4

# Utilities
numpy >= 1.24.0
tqdm >= 4.65.0
wandb >= 0.15.0  # Experiment tracking
tensorboard >= 2.13.0

# Data Handling
albumentations >= 1.3.0
datasets >= 2.14.0  # Hugging Face datasets
```

### 6.2 Hardware Configuration

**Minimum Requirements:**
- GPU: NVIDIA RTX 3050 (6GB VRAM)
- RAM: 16GB system memory
- Storage: 100GB SSD (for datasets and checkpoints)
- CPU: 4+ cores for data loading

**Memory Optimization Strategies:**
- Mixed precision training (FP16) reduces VRAM by ~40%[11]
- Gradient accumulation simulates larger batch sizes[11]
- Checkpoint gradients in generator to trade compute for memory[16]

**Expected Training Time:**
- Pretraining: 6 hours
- GAN training: 4 hours
- **Total: ~10 hours**[11]

### 6.3 Project Structure

```
VintageGAN/
├── data/
│   ├── imagenet_subset/       # 10k clean images
│   ├── filmset/                # Real vintage photos for FID
│   ├── synthetic_pairs/        # Generated training data
│   └── validation/             # Hold-out test set
├── models/
│   ├── generator.py            # U-Net + conditioning
│   ├── discriminator.py        # PatchGAN
│   ├── attention.py            # Self-attention module
│   └── detectors.py            # Defect extractors
├── training/
│   ├── pretrain.py             # Phase 1 training
│   ├── gan_train.py            # Phase 3 training
│   ├── losses.py               # Loss functions
│   └── dataset.py              # Data loaders
├── defects/
│   ├── grain.py                # Film grain synthesis
│   ├── scratches.py            # Scratch generation
│   ├── dust.py                 # Dust particles
│   ├── vignette.py             # Vignetting
│   ├── color_shift.py          # Color degradation
│   └── blur.py                 # Motion/lens blur
├── evaluation/
│   ├── metrics.py              # FID, SSIM, PSNR
│   ├── ablation.py             # Ablation experiments
│   └── user_study.py           # Perceptual evaluation
├── inference/
│   ├── apply_filter.py         # Single image inference
│   └── batch_process.py        # Folder processing
├── notebooks/
│   ├── demo.ipynb              # Interactive demo
│   └── analysis.ipynb          # Results visualization
├── configs/
│   └── training_config.yaml    # Hyperparameters
├── checkpoints/                # Saved models
├── logs/                       # TensorBoard logs
├── outputs/                    # Generated samples
├── requirements.txt
└── README.md
```

### 6.4 Code Quality Standards

**Enforce with Claude Sonnet 4.5:**
- Type hints for all function signatures
- Docstrings (Google style) for all classes and functions
- Maximum line length: 100 characters
- Use `black` formatter for consistency
- Add comments for non-obvious operations
- Modular functions (single responsibility principle)

**Testing:**
- Unit tests for each defect synthesis function
- Integration test for full pipeline
- Validate output shapes at each layer

***

## 7. ACADEMIC DOCUMENTATION

### 7.1 Report Structure[10][9]

**Title Page:**
- Project title: "VintageGAN: Parametric Controllable Vintage Film Defect Synthesis Using Conditional Generative Adversarial Networks"
- Student name, roll number, department
- Guide name and designation
- University logo and affiliation
- Submission date

**Preliminary Pages:**
- Certificate (guide signature)
- Declaration of originality
- Acknowledgements
- Abstract (200-250 words)
- Table of contents
- List of figures
- List of tables
- List of abbreviations (GAN, FID, SSIM, etc.)

**Chapter 1: Introduction (5-6 pages)**
- Background on film photography and digital imaging
- Problem statement: Need for controllable vintage effects
- Objectives (primary and secondary)
- Scope and limitations
- Organization of report

**Chapter 2: Literature Review (8-10 pages)**
- Section 2.1: Image-to-Image Translation (Pix2Pix, CycleGAN)
- Section 2.2: Conditional GANs and ControlNet
- Section 2.3: Film Restoration and Defect Analysis (cite )[14][19][21]
- Section 2.4: Style Transfer Techniques
- Section 2.5: Research Gap Identification
- Cite 25-30 papers with IEEE format[10]

**Chapter 3: Methodology (10-12 pages)**
- Section 3.1: System Architecture (block diagram)
- Section 3.2: Generator Design (architecture diagram, equations)
- Section 3.3: Discriminator Design
- Section 3.4: Conditioning Mechanism (mathematical formulation)
- Section 3.5: Defect Synthesis Algorithms (pseudocode for each)
- Section 3.6: Training Strategy (NoGAN phases)
- Section 3.7: Loss Functions (LaTeX equations)

**Chapter 4: Implementation (6-8 pages)**
- Section 4.1: Software and Hardware Setup
- Section 4.2: Dataset Creation Process
- Section 4.3: Training Procedure
- Section 4.4: Hyperparameter Tuning
- Section 4.5: Optimization Techniques (mixed precision, etc.)

**Chapter 5: Results and Discussion (8-10 pages)**
- Section 5.1: Quantitative Evaluation (tables with FID, SSIM, PSNR)
- Section 5.2: Ablation Study Results (comparison table)
- Section 5.3: Baseline Comparisons (graphs and tables)
- Section 5.4: Qualitative Analysis (image grids showing outputs)
- Section 5.5: User Study Results (statistical analysis)
- Section 5.6: Discussion of Findings
- Section 5.7: Limitations

**Chapter 6: Conclusion and Future Work (2-3 pages)**
- Summary of achievements
- Contributions to computer vision
- Future research directions (video extension, diffusion models)

**References (3-4 pages)**
- Minimum 30 IEEE-format citations

**Appendices**
- Appendix A: Code snippets (key functions)
- Appendix B: User study questionnaire
- Appendix C: Additional result images

**Page Count:** 40-50 pages[9][10]

### 7.2 Key Equations to Include

**Generator Loss (LaTeX):**
$$
\mathcal{L}_G = \mathcal{L}_{adv} + \lambda_1 \mathcal{L}_{pixel} + \lambda_2 \mathcal{L}_{perceptual} + \lambda_3 \mathcal{L}_{consistency}
$$

**Perceptual Loss:**
$$
\mathcal{L}_{perceptual} = \sum_{l \in \{conv3\_3, conv4\_3\}} \| \phi_l(G(x, c)) - \phi_l(y) \|_2^2
$$

**Consistency Loss:**
$$
\mathcal{L}_{consistency} = \| D_{extract}(G(x, c)) - c \|_2^2
$$

**FID Score:**
$$
FID = \| \mu_r - \mu_g \|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

### 7.3 Figures and Tables Required

**Figures:**
1. System architecture block diagram
2. Generator network architecture
3. Discriminator network architecture
4. Conditioning integration flowchart
5. Sample training pairs (grid of clean → defected)
6. Training loss curves (generator and discriminator)
7. FID score progression over epochs
8. Qualitative results grid (multiple defect levels)
9. Ablation study visual comparisons
10. Baseline method comparisons
11. User study interface screenshot

**Tables:**
1. Dataset statistics
2. Hyperparameter settings
3. Quantitative results (FID, SSIM, PSNR)
4. Ablation study numerical results
5. Baseline comparison metrics
6. User study statistical analysis
7. Computational requirements
8. Training time breakdown

***

## 8. VALIDATION CHECKLIST

### 8.1 Technical Validation

- [ ] Generator produces 512×512 RGB images
- [ ] All 6 condition parameters independently controllable
- [ ] Condition values smoothly interpolate (0.0 to 1.0)
- [ ] Model runs inference in < 2 seconds on RTX 3050
- [ ] Training completes in < 12 hours total
- [ ] FID score < 50 on validation set
- [ ] SSIM > 0.75 on validation set
- [ ] PSNR > 22 dB on validation set
- [ ] Condition accuracy MAE < 0.15
- [ ] No mode collapse (diverse outputs for same conditions)

### 8.2 Academic Validation

- [ ] Report follows university B.Tech format[9][10]
- [ ] Abstract is 200-250 words
- [ ] Literature review cites 25+ papers
- [ ] All equations properly formatted in LaTeX
- [ ] All figures have captions and are referenced in text
- [ ] Plagiarism < 15% (Turnitin check)
- [ ] Code on GitHub with proper README
- [ ] Presentation slides prepared (15-20 minutes)
- [ ] Demo video recorded (< 5 minutes)
- [ ] User study completed with 25+ participants

### 8.3 Code Quality Validation

- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Code passes `black` formatting
- [ ] No hardcoded paths (use config files)
- [ ] Reproducible with fixed random seeds
- [ ] Unit tests for defect synthesis functions
- [ ] Requirements.txt includes all dependencies
- [ ] README has installation and usage instructions

---

## 9. INCREMENTAL DEVELOPMENT PLAN FOR CLAUDE SONNET 4.5

### Day 1-2: Project Setup and Data Pipeline

**Tasks:**
1. Create project directory structure
2. Set up virtual environment and install dependencies
3. Download ImageNet subset (10k images)
4. Implement `data/dataset.py` with data loaders
5. Write unit tests for data loading

**Validation:**
- Can load and visualize ImageNet samples
- Data loader returns correct tensor shapes: (batch, 3, 512, 512)

### Day 3-4: Defect Synthesis Implementation

**Tasks:**
1. Implement all 6 defect functions in `defects/` directory
2. Create `defects/combined.py` to apply multiple defects
3. Generate 1000 sample training pairs
4. Visualize defect progressions (0.0, 0.3, 0.6, 1.0 intensity)

**Validation:**
- Each defect function works independently
- Combined defects look realistic
- Condition vectors correctly map to visual outputs

### Day 5-6: Generator Architecture

**Tasks:**
1. Implement `models/generator.py` with U-Net
2. Implement `models/attention.py` for self-attention
3. Implement conditioning integration (spatial replication + MLP)
4. Test forward pass with dummy input

**Validation:**
- Generator outputs (batch, 3, 512, 512) tensor
- Parameters count ~50-70M (reasonable size)
- Forward pass completes in < 0.5 seconds

### Day 7: Discriminator Architecture

**Tasks:**
1. Implement `models/discriminator.py` with PatchGAN
2. Add spectral normalization
3. Test forward pass with real/fake images

**Validation:**
- Discriminator outputs (batch, 32, 32, 1) predictions
- Accepts conditioned inputs correctly

### Day 8-9: Pretraining Phase

**Tasks:**
1. Implement `training/losses.py` with perceptual and pixel losses
2. Implement `training/pretrain.py` with training loop
3. Set up TensorBoard logging
4. Train for 40 epochs

**Validation:**
- Pretraining loss decreases consistently
- Generated images visually similar to targets
- Checkpoint saving works correctly

### Day 10: Defect Detectors

**Tasks:**
1. Implement 6 detector networks in `models/detectors.py`
2. Train each detector on synthetic dataset
3. Validate detector accuracy (MAE < 0.10 on test set)

**Validation:**
- Detectors correctly predict defect intensities
- Lightweight models (< 10M parameters each)

### Day 11-12: GAN Training

**Tasks:**
1. Implement `training/gan_train.py` with adversarial + consistency losses
2. Pretrain discriminator (10 epochs)
3. Fine-tune with GAN (10 epochs)
4. Monitor FID score every epoch

**Validation:**
- GAN training stable (no mode collapse)
- FID score improves during training
- Generated images look authentic

### Day 13: Evaluation Implementation

**Tasks:**
1. Implement `evaluation/metrics.py` with FID, SSIM, PSNR
2. Run ablation studies (5 variants)
3. Compare against baseline methods
4. Generate results tables and graphs

**Validation:**
- Metrics calculated correctly
- Full model outperforms ablated versions
- Results ready for report

### Day 14: Inference and Demo

**Tasks:**
1. Implement `inference/apply_filter.py` for single image processing
2. Create `notebooks/demo.ipynb` with interactive sliders
3. Process 50 test images with varied conditions
4. Record demo video

**Validation:**
- Inference works on arbitrary images
- Sliders smoothly control defect levels
- Demo is presentation-ready

### Day 15: Documentation and Cleanup

**Tasks:**
1. Write comprehensive README.md
2. Add docstrings to all functions
3. Format code with `black`
4. Prepare GitHub repository
5. Start report writing (use results from evaluation)

**Validation:**
- Code is clean and documented
- GitHub repo is public and organized
- Report outline complete

***

## 10. DEBUGGING GUIDELINES FOR CLAUDE

### 10.1 Common Issues and Solutions

**Issue:** CUDA out of memory during training
**Solution:**
- Reduce batch size from 8 to 4
- Enable gradient checkpointing in generator
- Use mixed precision (FP16)
- Clear cache with `torch.cuda.empty_cache()`

**Issue:** Mode collapse (generator produces same output)
**Solution:**
- Increase discriminator training frequency (2:1 ratio)
- Add spectral normalization to both G and D
- Use label smoothing (0.9 instead of 1.0 for real labels)
- Reduce GAN training epochs (stop at 5 if collapse occurs)

**Issue:** FID score not improving
**Solution:**
- Ensure FilmSet reference dataset is correct
- Increase pretraining epochs (40 → 60)
- Add more synthetic training pairs (50k → 100k)
- Tune perceptual loss weight (0.1 → 0.2)

**Issue:** Conditions not respected (consistency loss high)
**Solution:**
- Verify detectors are trained correctly (test accuracy > 90%)
- Increase consistency loss weight (0.3 → 0.5)
- Generate more diverse condition combinations in training data
- Check condition vector normalization (must be )[30]

### 10.2 Incremental Testing Strategy

**After each implementation step:**
1. Run unit test for that component
2. Check tensor shapes with `print(tensor.shape)`
3. Visualize intermediate outputs (save as images)
4. Verify gradients are flowing (`requires_grad=True`)
5. Test on single batch before full training

**Never proceed to next step if current step has errors**

### 10.3 Critical Checkpoints

**Before starting GAN training:**
- Pretrained generator must produce recognizable images (visual check)
- Validation loss should be < 0.1 for pretraining
- All detectors should have MAE < 0.15 on test set

**Before evaluation:**
- Have at least 1000 generated validation images
- FilmSet dataset properly downloaded and organized
- All metrics functions tested on dummy data

**Before documentation:**
- All quantitative results finalized in tables
- All figures generated and saved as high-res PNGs
- Code is clean, commented, and formatted

***

## 11. SUCCESS CRITERIA SUMMARY

### 11.1 Technical Success

✅ **Functional Model:**
- Generates 512×512 vintage images with 6 controllable parameters
- Inference time < 2 seconds per image on RTX 3050
- Training completes in < 12 hours

✅ **Performance Metrics:**
- FID < 50 (vs real vintage photos)
- SSIM > 0.75 (structural preservation)
- PSNR > 22 dB (quality retention)
- Condition MAE < 0.15 (control accuracy)

✅ **Robustness:**
- Works on diverse image types (portraits, landscapes, objects)
- Smooth interpolation between defect levels
- No mode collapse or artifacts

### 11.2 Academic Success

✅ **Documentation:**
- 40-50 page report following B.Tech format[10][9]
- 25+ peer-reviewed citations in IEEE format
- All equations in LaTeX
- 10+ figures and 7+ tables

✅ **Validation:**
- Ablation study showing each component's value
- Baseline comparisons (3+ methods)
- User study with 25+ participants
- Statistical significance tests (p < 0.05)

✅ **Presentation:**
- 15-20 minute defense presentation
- Live demo or recorded video
- GitHub repository with clean code

### 11.3 Innovation and Impact

✅ **Novel Contributions:**
- Parametric control over 6 independent defect types[3][2]
- Consistency loss for condition adherence[23][12]
- Efficient NoGAN training strategy[11]
- Synthetic defect dataset and evaluation framework

✅ **Potential Publication:**
- Target: Regional CV conference or workshop (NCVPRIPG, CVPR workshop)
- Unique selling point: Multi-parameter controllability
- Academic angle: Bridging style transfer with parametric generation

***

## 12. RISK MITIGATION

### 12.1 Timeline Risks

**Risk:** Training takes longer than 12 hours
**Mitigation:**
- Start with smaller dataset (5k images) for rapid iteration
- Use pretraining-only model if GAN training fails
- Reduce image resolution to 256×256 temporarily

**Risk:** Claude generates buggy code requiring extensive debugging
**Mitigation:**
- Implement incrementally (test each component)
- Use unit tests to catch errors early
- Have fallback to simpler architectures if complex ones fail

### 12.2 Performance Risks

**Risk:** FID score doesn't reach target (< 50)
**Mitigation:**
- Extend pretraining epochs
- Use higher quality reference dataset (FilmSet)
- Accept FID < 60 as "good" for academic project

**Risk:** Conditions not accurately represented in outputs
**Mitigation:**
- Train detectors longer for higher accuracy
- Increase consistency loss weight
- Manually verify defect synthesis functions produce correct levels

### 12.3 Resource Risks

**Risk:** RTX 3050 6GB VRAM insufficient
**Mitigation:**
- Reduce batch size to 4 or 2
- Use gradient accumulation (effective batch size 32)
- Enable mixed precision (FP16)
- Use gradient checkpointing
- Consider Google Colab Pro as backup (T4/V100 GPUs)

***

## 13. FINAL DELIVERABLES CHECKLIST

### 13.1 Code Deliverables

- [ ] Complete GitHub repository with organized structure
- [ ] README.md with installation and usage instructions
- [ ] requirements.txt with pinned versions
- [ ] Pretrained model weights (upload to Google Drive/Hugging Face)
- [ ] Jupyter notebook demo with interactive sliders
- [ ] Batch processing script for folder of images
- [ ] Unit tests for all critical functions

### 13.2 Documentation Deliverables

- [ ] 40-50 page B.Tech project report (PDF)
- [ ] Plagiarism report showing < 15% similarity
- [ ] Presentation slides (PPT/PDF, 15-20 slides)
- [ ] Demo video (< 5 minutes, MP4)
- [ ] User study results and analysis (Excel/CSV)

### 13.3 Evaluation Deliverables

- [ ] Quantitative results table (FID, SSIM, PSNR, IS)
- [ ] Ablation study results table
- [ ] Baseline comparison table and graphs
- [ ] Sample output images (50+ examples)
- [ ] User study statistical analysis

### 13.4 Academic Deliverables

- [ ] Signed certificate from project guide
- [ ] Declaration of originality
- [ ] Bound hard copy of report (for university submission)
- [ ] Soft copy on CD/USB (if required by university)

***

## 14. POST-SUBMISSION ENHANCEMENTS (OPTIONAL)

### 14.1 Publication-Ready Extensions

1. **Expand to Video:** Add temporal consistency for video filtering[31][8]
2. **Diffusion Comparison:** Implement ControlNet-based diffusion model[1]
3. **Mobile Deployment:** Optimize with ONNX/TensorFlow Lite[32]
4. **Dataset Release:** Publish synthetic defect dataset on Hugging Face[3]
5. **Web Demo:** Deploy Gradio/Streamlit app on Hugging Face Spaces

### 14.2 Startup Potential

1. **API Development:** REST API for vintage filter application
2. **Plugin Creation:** Figma/Photoshop/Lightroom plugins[33]
3. **Mobile App:** iOS/Android app using TFLite model
4. **SaaS Platform:** Subscription-based vintage photo processing service

---

**END OF SPECIFICATION DOCUMENT**

**Total Pages:** 14  
**Total Words:** ~8,500  
**Estimated Reading Time:** 30 minutes  

**Instructions for Claude Sonnet 4.5 in Windsurf:**
1. Read this entire document before writing any code
2. Implement components in the order specified in Section 9 (Days 1-15)
3. After each day's tasks, validate against the checklist in that section
4. If any step fails validation, debug before proceeding
5. Make only small adjustments unless explicitly required
6. Document all deviations from this spec in comments
7. Prioritize correctness over speed - no shortcuts that compromise quality

This specification is designed to be **self-contained and complete**. Follow it precisely for reproducible results.[34][25][16]