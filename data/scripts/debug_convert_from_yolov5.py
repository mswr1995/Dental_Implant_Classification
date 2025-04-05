import os
import shutil
import yaml
import glob
from pathlib import Path
import re
import sys
import random
from collections import Counter

# At the top of your script
MATCHING_STRICTNESS = 'strict'  # Options: 'strict', 'moderate', 'loose'

# Valid dental implant brands we want to recognize
VALID_CLASSES = {
    'Bego', 'Bicon', 'ITI', 'ADIN', 'DIONAVI', 
    'Dentium', 'MIS', 'NORIS', 'nobel', 'osstem'
}

# Mapping variations of class names to our standardized names
CLASS_MAPPINGS = {
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
    'NORIS': 'NORIS',
    'class 0': None,  # These will be handled by filename analysis
    'class 1': None,
    'class 2': None,
    'class 3': None,
    'class 4': None,
    'class 5': None,
    'class 6': None,
    'class 7': None,
    'class 8': None,
    'class 9': None
}

def read_yaml_classes(yaml_path):
    """Read class names from YAML file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data.get('names', {})

def normalize_class_name(name):
    if not name:
        return None
    
    # Remove any leading/trailing whitespace and quotes
    name = name.strip().strip('"\'')
    
    # Check direct mapping first (safest)
    if name in CLASS_MAPPINGS:
        return CLASS_MAPPINGS[name]
    
    # Word boundary matching instead of substring matching
    for valid_class in VALID_CLASSES:
        if re.search(r'\b' + re.escape(valid_class.lower()) + r'\b', name.lower()):
            return valid_class
            
    return None

def extract_class_from_yolo_txt(txt_path, class_names):
    """Extract class information from YOLO annotation file."""
    classes = set()
    try:
        with open(txt_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id < len(class_names):
                        raw_class = class_names[class_id]
                        normalized_class = normalize_class_name(raw_class)
                        if normalized_class:
                            classes.add(normalized_class)
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
    return list(classes)

def extract_class_from_filename(filename, class_names=None):
    """Try to extract class from filename based on valid classes."""
    filename_lower = filename.lower()
    
    # Split filename into parts based on common separators
    name_parts = re.split(r'[-_\s.]', filename_lower)
    
    # First try exact matches with name parts
    for part in name_parts:
        part = part.strip()
        if not part:
            continue
            
        # Check for exact matches with valid classes or known variations
        for class_name in VALID_CLASSES:
            if class_name.lower() == part:
                return [class_name]
        
        for key, val in CLASS_MAPPINGS.items():
            if val and key.lower() == part:
                return [val]
    
    # Only if exact matching fails, try more careful partial matching
    # This is safer with special cases like brand names that are distinctive
    special_cases = {
        'straumann': 'ITI',
        'biocare': 'nobel'
    }
    
    for special, class_name in special_cases.items():
        if special in filename_lower:
            return [class_name]
    
    # As a last resort, check for word boundaries using regex
    # This prevents matching substrings within other words
    for class_name in VALID_CLASSES:
        class_lower = class_name.lower()
        # Use word boundary check to prevent matching substrings
        if re.search(r'\b' + re.escape(class_lower) + r'\b', filename_lower):
            return [class_name]
    
    return []

def get_best_class_match(filename, annotations=None):
    """Get the best class match using multiple strategies with confidence scores."""
    matches = []
    
    # Try annotation first (highest confidence)
    if annotations:
        for class_id, conf in annotations:
            if class_id in CLASS_MAPPINGS:
                matches.append((CLASS_MAPPINGS[class_id], 1.0))
    
    # Try exact word matches from filename (high confidence)
    name_parts = re.split(r'[-_\s.]', filename.lower())
    for part in name_parts:
        for class_name in VALID_CLASSES:
            if class_name.lower() == part:
                matches.append((class_name, 0.9))
    
    # More matching strategies with decreasing confidence...
    
    # Return best match if any found
    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]
    
    return None

def convert_dataset(dataset_path, output_base_path, create_val=True, val_split=0.2):
    """Convert a YOLOv5 dataset to classification format."""
    # Get dataset name
    dataset_name = os.path.basename(dataset_path)
    output_path = os.path.join(output_base_path, dataset_name)
    
    print(f"\n=== Processing dataset: {dataset_name} ===")
    
    # Read class names from data.yaml
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    raw_class_names = []
    
    if os.path.exists(yaml_path):
        raw_class_names = read_yaml_classes(yaml_path)
        print(f"Found YAML class definitions: {raw_class_names}")
    else:
        print(f"Warning: No data.yaml found in {dataset_path}, will rely on filename detection only")
    
    # Create normalized class mapping dictionary
    class_mapping = {}
    for i, raw_name in enumerate(raw_class_names):
        normalized = normalize_class_name(raw_name)
        if normalized:
            class_mapping[i] = normalized
    
    # If no valid classes were found in YAML, still proceed but rely on filename detection only
    if not class_mapping and raw_class_names:
        print(f"Warning: None of the classes in YAML could be mapped to valid classes.")
        print(f"Will attempt to extract classes from filenames instead.")
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    # Check for validation set
    has_val = os.path.exists(os.path.join(dataset_path, 'val', 'images'))
    
    # Track all processed images and their classes
    processed_images = {'train': [], 'val': [], 'test': []}
    class_counter = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    
    # Process all splits - first collect all images with their detected classes
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(dataset_path, split, 'images')
        if not os.path.exists(split_images_dir):
            print(f"Split {split} not found in {dataset_name}")
            continue
            
        print(f"Processing {split} split...")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(split_images_dir, ext)))
        
        print(f"Found {len(image_files)} images in {split} split")
        
        for img_path in image_files:
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            
            # Try to find corresponding label file
            label_path = os.path.join(dataset_path, split, 'labels', f"{img_name}.txt")
            classes = []
            
            # First try to get class from annotation file if it exists
            if os.path.exists(label_path) and raw_class_names:
                classes = extract_class_from_yolo_txt(label_path, raw_class_names)
            
            # If no classes found from annotation, try filename
            if not classes:
                classes = extract_class_from_filename(img_name)
                if classes:
                    print(f"Extracted class from filename: {img_name} -> {classes}")
            
            # If still no classes, no aggressive matching - we only want exact matches
            if not classes:
                print(f"Could not find a class match for {img_name}")
            
            # If we found classes, add to processed images
            if classes:
                processed_images[split].append((img_path, classes))
                for cls in classes:
                    class_counter[split][cls] += 1
            else:
                print(f"Warning: Could not determine class for {img_path}")
    
    # Create output directories for each split and class
    for split in ['train', 'val', 'test']:
        split_output_dir = os.path.join(output_path, split)
        for class_name in VALID_CLASSES:
            os.makedirs(os.path.join(split_output_dir, class_name), exist_ok=True)
    
    # Print initial class distribution
    for split in ['train', 'val', 'test']:
        if class_counter[split]:
            print(f"Initial {split} class distribution: {dict(class_counter[split])}")
    
    # Handle validation split creation if needed
    if create_val and not has_val and processed_images['train']:
        print(f"Creating validation set from training data (split: {val_split})")
        
        # Group images by class to ensure balanced split
        images_by_class = {}
        for img_path, classes in processed_images['train']:
            for cls in classes:
                if cls not in images_by_class:
                    images_by_class[cls] = []
                images_by_class[cls].append((img_path, [cls]))
        
        # Split each class separately
        val_images = []
        train_images = []
        
        for cls, imgs in images_by_class.items():
            random.shuffle(imgs)
            val_size = max(1, int(len(imgs) * val_split))
            val_images.extend(imgs[:val_size])
            train_images.extend(imgs[val_size:])
        
        # Update the processed images
        processed_images['train'] = train_images
        processed_images['val'] = val_images
        
        # Recalculate counters
        class_counter['train'] = Counter()
        class_counter['val'] = Counter()
        
        for _, classes in train_images:
            for cls in classes:
                class_counter['train'][cls] += 1
                
        for _, classes in val_images:
            for cls in classes:
                class_counter['val'][cls] += 1
        
        print(f"After split - Train class distribution: {dict(class_counter['train'])}")
        print(f"After split - Val class distribution: {dict(class_counter['val'])}")
    
    # Process and copy images for each split
    skipped_count = 0
    copied_count = 0
    
    for split in ['train', 'val', 'test']:
        if not processed_images[split]:
            continue
            
        print(f"Copying {len(processed_images[split])} images to {split} split...")
        
        for img_path, classes in processed_images[split]:
            img_filename = os.path.basename(img_path)
            valid_classes = [c for c in classes if c in VALID_CLASSES]
            
            if not valid_classes:
                skipped_count += 1
                continue
                
            for class_name in valid_classes:
                dest_path = os.path.join(output_path, split, class_name, img_filename)
                shutil.copy(img_path, dest_path)
                copied_count += 1
    
    print(f"Finished processing {dataset_name}: copied {copied_count} images, skipped {skipped_count} images")
    
    # Check if we actually copied any images
    if copied_count == 0:
        print(f"WARNING: No images were copied for dataset {dataset_name}. Check if class names in YAML match our VALID_CLASSES!")
        # List some sample image filenames to help debug
        for split in ['train', 'val', 'test']:
            split_images_dir = os.path.join(dataset_path, split, 'images')
            if os.path.exists(split_images_dir):
                sample_images = glob.glob(os.path.join(split_images_dir, '*.jpg'))[:5]
                if sample_images:
                    print(f"Sample {split} images: {[os.path.basename(img) for img in sample_images]}")

def main():
    # Get the absolute path to the project root from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
    
    # Define paths relative to project root
    data_collected_dir = os.path.join(project_root, 'data', 'data_collected')
    output_base_dir = os.path.join(project_root, 'data', 'classification_datasets')
    
    # Check if data_collected directory exists
    if not os.path.exists(data_collected_dir):
        print(f"Error: Directory not found: {data_collected_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Available directories in data/: {os.listdir(os.path.dirname(data_collected_dir))}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Find all dataset directories
    datasets = [d for d in os.listdir(data_collected_dir) 
                if os.path.isdir(os.path.join(data_collected_dir, d))]
    
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    # Process each dataset
    for dataset in datasets:
        dataset_path = os.path.join(data_collected_dir, dataset)
        convert_dataset(dataset_path, output_base_dir, create_val=True, val_split=0.2)
    
    # Print dataset statistics
    print_dataset_stats(output_base_dir)
    
    print("Conversion complete!")

def print_dataset_stats(base_dir):
    """Print statistics about the dataset conversion."""
    total_stats = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        
        print(f"\nDataset: {dataset_dir}")
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue
                
            print(f"  {split.capitalize()} split:")
            split_total = 0
            
            for class_name in VALID_CLASSES:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                    total_stats[split][class_name] += count
                    split_total += count
                    print(f"    {class_name}: {count} images")
            
            print(f"    Total: {split_total} images")
    
    print("\n--- OVERALL STATISTICS ---")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.capitalize()} split:")
        split_total = sum(total_stats[split].values())
        for class_name in VALID_CLASSES:
            count = total_stats[split][class_name]
            percentage = (count / split_total * 100) if split_total > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        print(f"  Total: {split_total} images")

if __name__ == "__main__":
    main()