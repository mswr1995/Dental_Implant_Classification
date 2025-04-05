import os
import cv2
import numpy as np
from pathlib import Path
import random
import logging
from PIL import Image
import imagehash
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt  # For debugging visualizations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DuplicateInfo:
    def __init__(self):
        self.duplicates_found = 0
        self.duplicate_groups = defaultdict(list)
        self.kept_images = set()
        self.removed_images = set()
        self.hash_size = 12  # MUCH LARGER hash size for dental X-rays (was too small at 4)

class DataProcessor:
    def __init__(self, raw_data_path: str, output_path: str, image_size: int = 512, debug=False):
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.debug = debug
        
        if not self.raw_data_path.is_absolute():
            self.raw_data_path = self.project_root / self.raw_data_path
        if not self.output_path.is_absolute():
            self.output_path = self.project_root / self.output_path
            
        self.image_size = image_size
        self.duplicate_info = DuplicateInfo()
        
        self.stats = {
            'processed_images': 0,
            'failed_images': 0,
            'class_distribution': defaultdict(int),
            'split_distribution': defaultdict(int),
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'original_size_stats': defaultdict(int)
        }
        
        # Create debug directory if needed
        if self.debug:
            self.debug_dir = self.project_root / 'data/debug'
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def apply_radiographic_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply specialized enhancements for dental radiographs with safer parameters."""
        # Make a copy to avoid modifying the original
        result = image.copy()
        
        # 1. Apply a milder bilateral filter
        # Less aggressive parameters to preserve more detail
        denoised = cv2.bilateralFilter(result, 3, 25, 25)
        
        # 2. Apply gentler CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Apply a gentler sharpening
        kernel = np.array([[-0.5, -0.5, -0.5], 
                           [-0.5,  5.0, -0.5], 
                           [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Debug: Save intermediate steps if needed
        if self.debug and random.random() < 0.01:  # Save 1% of images for debugging
            debug_path = self.debug_dir / f"debug_{random.randint(1000, 9999)}"
            debug_path.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_path / "1_original.jpg"), image)
            cv2.imwrite(str(debug_path / "2_denoised.jpg"), denoised)
            cv2.imwrite(str(debug_path / "3_enhanced.jpg"), enhanced)
            cv2.imwrite(str(debug_path / "4_sharpened.jpg"), sharpened)
        
        return sharpened

    def process_image(self, image_path: Path) -> Tuple[np.ndarray, dict]:
        """Process a single image with radiographic-specific enhancements."""
        # Read image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Record original dimensions for statistics
        h, w = image.shape
        metadata = {'original_height': h, 'original_width': w}
        self.stats['original_size_stats'][f"{w}x{h}"] += 1
        
        # Calculate padding to maintain aspect ratio
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize maintaining aspect ratio with high-quality interpolation
        if scale < 1:  # Downsampling
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:  # Upsampling
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create padded image (use 128 as padding value instead of 0 for better visualization)
        padded = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 128
        h_offset = (self.image_size - new_h) // 2
        w_offset = (self.image_size - new_w) // 2
        padded[h_offset:h_offset+new_h, w_offset:w_offset+new_w] = resized
        
        # Apply radiographic-specific enhancements
        enhanced = self.apply_radiographic_enhancement(padded)
        
        # Ensure proper normalization - prevent black images by checking min/max
        min_val, max_val = enhanced.min(), enhanced.max()
        if max_val > min_val:  # Only normalize if there's a range of values
            enhanced = np.clip(((enhanced - min_val) * 255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
        
        # If we somehow still have a black image, use the original padded version
        if enhanced.mean() < 10:  # Very dark image
            logger.warning(f"Enhancement produced very dark image for {image_path.name}, using original")
            enhanced = padded
        
        return enhanced, metadata

    def detect_duplicates(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """Detect duplicates using perceptual hashing with dental X-ray specific settings."""
        hash_dict = {}
        duplicate_groups = {}
        logger.info("Starting duplicate detection with dental X-ray specific settings...")
        
        # Much stricter threshold for dental X-rays which naturally look similar
        threshold = 2  # Reduced from 5 to be much stricter
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'L':
                        img = img.convert('L')
                    # Use a combination of perceptual hash and difference hash for more accuracy
                    p_hash = imagehash.phash(img, hash_size=self.duplicate_info.hash_size)
                    d_hash = imagehash.dhash(img, hash_size=self.duplicate_info.hash_size)
                    # Create a combined hash string
                    img_hash_str = f"{p_hash}_{d_hash}"
                
                # Compare with existing hashes - exact match only for dental X-rays
                found_duplicate = False
                for existing_hash_str, paths in hash_dict.items():
                    if existing_hash_str == img_hash_str:  # Exact match required
                        if existing_hash_str not in duplicate_groups:
                            duplicate_groups[existing_hash_str] = paths.copy()
                        
                        duplicate_groups[existing_hash_str].append(img_path)
                        self.duplicate_info.duplicates_found += 1
                        logger.info(f"Found exact duplicate: {img_path.name}")
                        found_duplicate = True
                        break
                
                if not found_duplicate:
                    hash_dict[img_hash_str] = [img_path]
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Additional validation - reject groups where the images are from different classes
        valid_duplicate_groups = {}
        for hash_str, group in duplicate_groups.items():
            # Extract class from path (assuming class is parent directory name)
            classes = set(path.parent.name for path in group)
            if len(classes) == 1:  # All images in group are from same class
                valid_duplicate_groups[hash_str] = group
            else:
                logger.warning(f"Rejected duplicate group with mixed classes: {classes}")
        
        logger.info(f"Found {len(valid_duplicate_groups)} valid duplicate groups with {sum(len(g)-1 for g in valid_duplicate_groups.values())} duplicates")
        return valid_duplicate_groups

    def _are_similar_filenames(self, path1: Path, path2: Path) -> bool:
        """Always returns True to disable filename-based filtering."""
        return True

    def create_splits(self):
        """Create train/val/test splits."""
        train_ratio, val_ratio = 0.7, 0.15  # test = 0.15
        
        for class_dir in self.raw_data_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            # Get all valid images (excluding duplicates)
            images = [p for p in class_dir.glob('*.jpg') 
                     if p not in self.duplicate_info.removed_images]
            
            if not images:
                continue
                
            # Create splits
            random.shuffle(images)
            n = len(images)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            splits = {
                'train': images[:train_idx],
                'val': images[train_idx:val_idx],
                'test': images[val_idx:]
            }
            
            # Process and save each split
            for split_name, split_images in splits.items():
                self._process_split(split_name, split_images, class_dir.name)

    def _process_split(self, split_name: str, image_paths: List[Path], class_name: str):
        """Process and save images for a specific split."""
        output_dir = self.output_path / split_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_paths:
            try:
                processed_image, metadata = self.process_image(img_path)
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), processed_image)
                
                self.stats['processed_images'] += 1
                self.stats['split_distribution'][split_name] += 1
                self.stats['class_distribution'][class_name] += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                self.stats['failed_images'] += 1

    def process_dataset(self):
        """Main processing method."""
        try:
            logger.info(f"Starting dataset processing with image size {self.image_size}x{self.image_size}...")
            
            # Collect all image paths
            all_images = []
            for class_dir in self.raw_data_path.iterdir():
                if class_dir.is_dir():
                    img_list = list(class_dir.glob('*.jpg'))
                    logger.info(f"Found {len(img_list)} images in {class_dir.name}")
                    all_images.extend(img_list)
            
            if not all_images:
                logger.warning("No images found in the dataset directory!")
                return
                
            logger.info(f"Found total of {len(all_images)} images across all classes")
            
            # Detect and handle duplicates
            duplicate_groups = self.detect_duplicates(all_images)
            
            # Keep only one image from each duplicate group
            for _, group in duplicate_groups.items():
                if len(group) > 1:  # Only process actual groups with duplicates
                    # Choose the image with the highest quality (using file size as proxy)
                    best_image = max(group, key=lambda x: x.stat().st_size)
                    self.duplicate_info.kept_images.add(best_image)
                    self.duplicate_info.removed_images.update(set(group) - {best_image})
                    logger.info(f"Keeping {best_image.name}, removing {len(group)-1} duplicates")
            
            # Output duplicate information
            logger.info(f"Found {self.duplicate_info.duplicates_found} duplicate images")
            logger.info(f"Keeping {len(self.duplicate_info.kept_images)} unique images")
            logger.info(f"Removing {len(self.duplicate_info.removed_images)} duplicate images")
            
            # Create directory structure and process images
            self.create_splits()
            self._log_statistics()
            
            logger.info("Dataset processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during dataset processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _log_statistics(self):
        """Log processing statistics."""
        logger.info("\n=== Processing Statistics ===")
        logger.info(f"Total images processed: {self.stats['processed_images']}")
        logger.info(f"Failed images: {self.stats['failed_images']}")
        logger.info(f"Duplicates found: {self.duplicate_info.duplicates_found}")
        logger.info(f"Duplicates removed: {len(self.duplicate_info.removed_images)}")
        
        logger.info("\nClass Distribution:")
        for class_name, count in self.stats['class_distribution'].items():
            logger.info(f"{class_name}: {count}")
        
        logger.info("\nSplit Distribution:")
        for split_name, count in self.stats['split_distribution'].items():
            logger.info(f"{split_name}: {count}")

def main():
    processor = DataProcessor(
        raw_data_path='data/data_raw',
        output_path='data/data_processed',
        image_size=512,
        debug=True  # Enable debugging
    )
    processor.process_dataset()

if __name__ == "__main__":
    main()
