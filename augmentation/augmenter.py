"""
Image Augmentation Pipeline
Applies realistic distortions to invoice images
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline"""
    # Noise
    add_noise: bool = True
    noise_probability: float = 0.5
    noise_intensity: Tuple[float, float] = (0.01, 0.05)
    
    # Blur
    add_blur: bool = True
    blur_probability: float = 0.3
    blur_kernel_size: Tuple[int, int] = (3, 7)
    
    # Compression artifacts
    add_compression: bool = True
    compression_probability: float = 0.4
    jpeg_quality: Tuple[int, int] = (60, 95)
    
    # Rotation/Deskew
    add_rotation: bool = True
    rotation_probability: float = 0.6
    rotation_angle: Tuple[float, float] = (-5.0, 5.0)
    
    # Perspective transform
    add_perspective: bool = True
    perspective_probability: float = 0.3
    perspective_strength: Tuple[float, float] = (0.01, 0.05)
    
    # Brightness/Contrast
    adjust_brightness: bool = True
    brightness_probability: float = 0.5
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    
    adjust_contrast: bool = True
    contrast_probability: float = 0.5
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    # Stains and marks
    add_stains: bool = True
    stain_probability: float = 0.2
    stain_count: Tuple[int, int] = (1, 3)
    
    # Shadows
    add_shadow: bool = True
    shadow_probability: float = 0.3
    
    # Folds/Creases
    add_crease: bool = True
    crease_probability: float = 0.2


class ImageAugmenter:
    """Applies augmentations to images"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter
        
        Args:
            config: AugmentationConfig or None for defaults
        """
        self.config = config or AugmentationConfig()
    
    def add_gaussian_noise(self, image: np.ndarray, intensity: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.randn(*image.shape) * intensity * 255
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image: np.ndarray, amount: float = 0.01) -> np.ndarray:
        """Add salt and pepper noise"""
        output = image.copy()
        
        # Salt
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        output[coords[0], coords[1]] = 255
        
        # Pepper
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        output[coords[0], coords[1]] = 0
        
        return output
    
    def apply_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_motion_blur(self, image: np.ndarray, size: int = 15, angle: float = 45) -> np.ndarray:
        """Apply motion blur"""
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Rotate kernel
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def apply_compression(self, image: np.ndarray, quality: int = 85) -> np.ndarray:
        """Simulate JPEG compression artifacts"""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save to bytes with compression
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load back
        compressed = Image.open(buffer)
        return cv2.cvtColor(np.array(compressed), cv2.COLOR_RGB2BGR)
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees)"""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust transformation matrix
        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation with white background
        rotated = cv2.warpAffine(
            image, M, (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return rotated
    
    def apply_perspective_transform(self, image: np.ndarray, strength: float = 0.03) -> np.ndarray:
        """Apply perspective transformation"""
        height, width = image.shape[:2]
        
        # Define source points (corners)
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        # Apply random offsets
        max_offset = int(min(width, height) * strength)
        offsets = np.random.randint(-max_offset, max_offset, size=(4, 2)).astype(np.float32)
        dst_points = src_points + offsets
        
        # Get transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(
            image, M, (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return transformed
    
    def adjust_brightness_contrast(self,
                                   image: np.ndarray,
                                   brightness: float = 1.0,
                                   contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast"""
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Adjust brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        # Adjust contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        # Convert back
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def add_stain(self, image: np.ndarray) -> np.ndarray:
        """Add realistic stain/mark"""
        height, width = image.shape[:2]
        output = image.copy()
        
        # Random position
        x = random.randint(0, width - 50)
        y = random.randint(0, height - 50)
        
        # Random size
        size = random.randint(20, 100)
        
        # Create stain shape (ellipse)
        stain = np.ones((size, size, 3), dtype=np.uint8) * 255
        color = random.randint(200, 240)
        cv2.ellipse(
            stain,
            (size//2, size//2),
            (size//2, size//3),
            random.randint(0, 360),
            0, 360,
            (color, color, color),
            -1
        )
        
        # Blur stain
        stain = cv2.GaussianBlur(stain, (21, 21), 0)
        
        # Blend with image
        alpha = random.uniform(0.1, 0.3)
        x_end = min(x + size, width)
        y_end = min(y + size, height)
        
        roi = output[y:y_end, x:x_end]
        stain_roi = stain[:y_end-y, :x_end-x]
        
        blended = cv2.addWeighted(roi, 1-alpha, stain_roi, alpha, 0)
        output[y:y_end, x:x_end] = blended
        
        return output
    
    def add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add shadow effect"""
        height, width = image.shape[:2]
        output = image.copy()
        
        # Create shadow mask
        shadow = np.ones((height, width), dtype=np.float32)
        
        # Random shadow parameters
        side = random.choice(['left', 'right', 'top', 'bottom'])
        intensity = random.uniform(0.3, 0.7)
        
        if side == 'left':
            for i in range(width // 3):
                shadow[:, i] = intensity + (1 - intensity) * (i / (width // 3))
        elif side == 'right':
            for i in range(width // 3):
                shadow[:, width - i - 1] = intensity + (1 - intensity) * (i / (width // 3))
        elif side == 'top':
            for i in range(height // 3):
                shadow[i, :] = intensity + (1 - intensity) * (i / (height // 3))
        else:  # bottom
            for i in range(height // 3):
                shadow[height - i - 1, :] = intensity + (1 - intensity) * (i / (height // 3))
        
        # Apply shadow
        for c in range(3):
            output[:, :, c] = (output[:, :, c] * shadow).astype(np.uint8)
        
        return output
    
    def add_crease(self, image: np.ndarray) -> np.ndarray:
        """Add fold/crease effect"""
        height, width = image.shape[:2]
        output = image.copy()
        
        # Random crease line
        if random.random() > 0.5:
            # Vertical crease
            x = random.randint(width // 4, 3 * width // 4)
            thickness = random.randint(2, 5)
            color = random.randint(180, 220)
            cv2.line(output, (x, 0), (x, height), (color, color, color), thickness)
        else:
            # Horizontal crease
            y = random.randint(height // 4, 3 * height // 4)
            thickness = random.randint(2, 5)
            color = random.randint(180, 220)
            cv2.line(output, (0, y), (width, y), (color, color, color), thickness)
        
        return output
    
    def augment(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply full augmentation pipeline
        
        Args:
            image: Input image (BGR format)
            seed: Random seed for reproducibility
            
        Returns:
            Augmented image
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        output = image.copy()
        
        # Noise
        if self.config.add_noise and random.random() < self.config.noise_probability:
            intensity = random.uniform(*self.config.noise_intensity)
            if random.random() > 0.5:
                output = self.add_gaussian_noise(output, intensity)
            else:
                output = self.add_salt_pepper_noise(output, intensity)
        
        # Blur
        if self.config.add_blur and random.random() < self.config.blur_probability:
            kernel_size = random.randrange(*self.config.blur_kernel_size, 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            if random.random() > 0.7:
                output = self.apply_motion_blur(output, kernel_size, random.uniform(0, 180))
            else:
                output = self.apply_blur(output, kernel_size)
        
        # Rotation
        if self.config.add_rotation and random.random() < self.config.rotation_probability:
            angle = random.uniform(*self.config.rotation_angle)
            output = self.rotate_image(output, angle)
        
        # Perspective
        if self.config.add_perspective and random.random() < self.config.perspective_probability:
            strength = random.uniform(*self.config.perspective_strength)
            output = self.apply_perspective_transform(output, strength)
        
        # Brightness/Contrast
        brightness = 1.0
        contrast = 1.0
        
        if self.config.adjust_brightness and random.random() < self.config.brightness_probability:
            brightness = random.uniform(*self.config.brightness_range)
        
        if self.config.adjust_contrast and random.random() < self.config.contrast_probability:
            contrast = random.uniform(*self.config.contrast_range)
        
        if brightness != 1.0 or contrast != 1.0:
            output = self.adjust_brightness_contrast(output, brightness, contrast)
        
        # Stains
        if self.config.add_stains and random.random() < self.config.stain_probability:
            num_stains = random.randint(*self.config.stain_count)
            for _ in range(num_stains):
                output = self.add_stain(output)
        
        # Shadow
        if self.config.add_shadow and random.random() < self.config.shadow_probability:
            output = self.add_shadow(output)
        
        # Crease
        if self.config.add_crease and random.random() < self.config.crease_probability:
            output = self.add_crease(output)
        
        # Compression (last step)
        if self.config.add_compression and random.random() < self.config.compression_probability:
            quality = random.randint(*self.config.jpeg_quality)
            output = self.apply_compression(output, quality)
        
        return output
    
    def augment_file(self, input_path: str, output_path: str, seed: Optional[int] = None):
        """
        Augment an image file
        
        Args:
            input_path: Path to input image
            output_path: Path to save augmented image
            seed: Random seed
        """
        image = cv2.imread(input_path)
        augmented = self.augment(image, seed)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), augmented)


class BatchAugmenter:
    """Batch augmentation of multiple images"""
    
    def __init__(self, augmenter: ImageAugmenter):
        """Initialize batch augmenter"""
        self.augmenter = augmenter
    
    def augment_batch(self,
                     input_paths: List[str],
                     output_dir: str,
                     copies_per_image: int = 1,
                     callback=None):
        """
        Augment multiple images
        
        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            copies_per_image: Number of augmented copies per image
            callback: Optional callback(index, total, filename)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total = len(input_paths) * copies_per_image
        current = 0
        
        for input_path in input_paths:
            input_file = Path(input_path)
            stem = input_file.stem
            ext = input_file.suffix
            
            for i in range(copies_per_image):
                current += 1
                
                if callback:
                    callback(current, total, f"{stem}_aug{i}{ext}")
                
                output_file = output_path / f"{stem}_aug{i}{ext}"
                self.augmenter.augment_file(str(input_path), str(output_file))


if __name__ == '__main__':
    # Example usage
    config = AugmentationConfig(
        noise_probability=0.7,
        blur_probability=0.5,
        rotation_probability=0.8
    )
    
    augmenter = ImageAugmenter(config)
    augmenter.augment_file(
        'data/raw/sample_invoice.png',
        'data/processed/sample_invoice_augmented.png'
    )
    
    print("Augmentation complete!")
