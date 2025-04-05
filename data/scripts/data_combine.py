import os
import shutil
from pathlib import Path
import yaml
import logging
from collections import Counter
from typing import Dict, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCombiner:
    def __init__(self, collected_data_path, output_path):
        # Get the project root directory (2 levels up from the script)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Convert relative paths to absolute paths
        self.collected_data_path = self.project_root / collected_data_path
        self.output_path = self.project_root / output_path
        
        logger.info(f"Data collection path: {self.collected_data_path}")
        logger.info(f"Output path: {self.output_path}")
        
        # Define valid classes and their variations
        self.valid_classes = {
            'Bego', 'Bicon', 'ITI', 'ADIN', 'DIONAVI', 
            'Dentium', 'MIS', 'NORIS', 'nobel', 'osstem'
        }
        
        # Enhanced class mappings with more variations
        self.class_mappings = {
            'nobel': 'nobel',
            'Nobel': 'nobel',
            'NOBEL': 'nobel',
            'Nobel Biocare': 'nobel',
            'osstem': 'osstem',
            'Osstem': 'osstem',
            'OSSTEM': 'osstem',
            'bego': 'Bego',
            'BEGO': 'Bego',
            'bicon': 'Bicon',
            'BICON': 'Bicon',
            'iti': 'ITI',
            'ITI': 'ITI',
            'straumann': 'ITI',  # ITI is also known as Straumann
            'Straumann': 'ITI',
            'adin': 'ADIN',
            'ADIN': 'ADIN',
            'dionavi': 'DIONAVI',
            'DIONAVI': 'DIONAVI',
            'dentium': 'Dentium',
            'DENTIUM': 'Dentium',
            'mis': 'MIS',
            'MIS': 'MIS',
            'noris': 'NORIS',
            'NORIS': 'NORIS'
        }
        
        # Track statistics
        self.dataset_stats = {
            'total_images': 0,
            'class_distribution': Counter(),
            'source_distribution': Counter(),
            'skipped_images': 0
        }

    def normalize_class_name(self, name):
        """Normalize class names to match valid_classes format."""
        if not name:
            return None
        
        # Remove any leading/trailing whitespace and quotes
        name = name.strip().strip('"\'')
        
        # Check direct mapping
        if name in self.class_mappings:
            return self.class_mappings[name]
        
        # Try capitalizing first letter
        capitalized = name.capitalize()
        if capitalized in self.valid_classes:
            return capitalized
            
        # Try upper case
        if name.upper() in self.valid_classes:
            return name.upper()
            
        # Try lower case
        if name.lower() in self.valid_classes:
            return name.lower()
            
        return None

    def get_class_from_filename(self, filename):
        """Extract class name from filename."""
        # Split by common separators and get the first part
        for separator in ['-', '_', ' ']:
            parts = filename.split(separator)
            if parts:
                normalized = self.normalize_class_name(parts[0])
                if normalized:
                    return normalized
        return None

    def read_yaml_config(self, yaml_path):
        """Read YAML configuration file."""
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_output_directories(self):
        """Create output directories for each class."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        for class_name in self.valid_classes:
            (self.output_path / class_name).mkdir(parents=True, exist_ok=True)

    def process_dataset(self, dataset_path):
        """Process a single dataset directory."""
        yaml_file = next(dataset_path.glob('*.yaml'), None)
        if not yaml_file:
            logger.warning(f"No YAML file found in {dataset_path}")
            return

        config = self.read_yaml_config(yaml_file)
        class_names = config.get('names', [])
        
        # Create mapping from numeric indices to class names
        class_mapping = {}
        for idx, name in enumerate(class_names):
            normalized = self.normalize_class_name(name)
            if normalized:
                class_mapping[str(idx)] = normalized

        # Process train/val/test directories
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split / 'images'
            if not split_dir.exists():
                continue

            labels_dir = dataset_path / split / 'labels'
            if not labels_dir.exists():
                continue

            self.copy_images_with_labels(split_dir, labels_dir, class_mapping)

    def copy_images_with_labels(self, images_dir, labels_dir, class_mapping):
        """Enhanced copy method with better tracking and validation."""
        processed = 0
        skipped = 0
        
        for image_file in images_dir.glob('*.*'):
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                skipped += 1
                continue

            # Get class name
            class_name = self.get_class_from_filename(image_file.stem)
            if not class_name:
                # Try to get class from label file
                label_file = labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            first_line = f.readline().strip()
                            if first_line:
                                class_idx = first_line.split()[0]
                                class_name = class_mapping.get(class_idx)
                    except Exception as e:
                        logger.warning(f"Error reading label file {label_file}: {e}")

            if class_name and class_name in self.valid_classes:
                dest_path = self.output_path / class_name / image_file.name
                
                # Check if destination already exists
                if dest_path.exists():
                    # Add a suffix to make the filename unique
                    counter = 1
                    while dest_path.exists():
                        new_name = f"{image_file.stem}_{counter}{image_file.suffix}"
                        dest_path = self.output_path / class_name / new_name
                        counter += 1

                # Copy the file
                shutil.copy2(image_file, dest_path)
                processed += 1
                
                # Update statistics
                self.dataset_stats['total_images'] += 1
                self.dataset_stats['class_distribution'][class_name] += 1
                self.dataset_stats['source_distribution'][images_dir.parent.parent.name] += 1
            else:
                skipped += 1
                self.dataset_stats['skipped_images'] += 1

        logger.info(f"Processed {processed} images, skipped {skipped} images in {images_dir}")

    def combine_datasets(self):
        """Main method to combine all datasets."""
        logger.info("Starting dataset combination process...")
        self.setup_output_directories()

        # Process each dataset directory
        for dataset_dir in self.collected_data_path.iterdir():
            if dataset_dir.is_dir():
                logger.info(f"Processing dataset: {dataset_dir}")
                self.process_dataset(dataset_dir)
                
        # Log comprehensive statistics
        self._log_statistics()

    def _log_statistics(self):
        """Log detailed statistics about the combined dataset."""
        logger.info("\n=== Dataset Combination Statistics ===")
        
        logger.info("\nClass Distribution:")
        for class_name, count in self.dataset_stats['class_distribution'].most_common():
            percentage = (count / self.dataset_stats['total_images']) * 100
            logger.info(f"{class_name}: {count} images ({percentage:.2f}%)")
        
        logger.info("\nSource Distribution:")
        for source, count in self.dataset_stats['source_distribution'].most_common():
            percentage = (count / self.dataset_stats['total_images']) * 100
            logger.info(f"{source}: {count} images ({percentage:.2f}%)")
        
        logger.info(f"\nTotal Images: {self.dataset_stats['total_images']}")
        logger.info(f"Skipped Images: {self.dataset_stats['skipped_images']}")
        
        # Check for class imbalance
        min_class_count = min(self.dataset_stats['class_distribution'].values())
        max_class_count = max(self.dataset_stats['class_distribution'].values())
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        logger.info(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            logger.warning("High class imbalance detected! Consider data augmentation or balancing techniques.")

def main():
    combiner = DatasetCombiner(
        collected_data_path='data/data_collected',
        output_path='data/data_raw'
    )
    combiner.combine_datasets()

if __name__ == "__main__":
    main()
