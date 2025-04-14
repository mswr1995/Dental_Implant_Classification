import os
import shutil
import yaml
import glob
from pathlib import Path
import re
import sys
import random
from collections import Counter
from PIL import Image, ImageFile
import copy

# Allow loading truncated images (use with caution)
ImageFile.LOAD_TRUNCATED_IMAGES = True

MATCHING_STRICTNESS = 'strict'  # Options: 'strict', 'moderate', 'loose'

VALID_CLASSES = {
    'Bego', 'Bicon', 'ITI', 'ADIN', 'DIONAVI',
    'Dentium', 'MIS', 'NORIS', 'nobel', 'osstem'
}

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
    'straumann': 'ITI',
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
    'class 0': None,
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
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data.get('names', {})

def normalize_class_name(name):
    if not name:
        return None
    
    name = name.strip().strip('"\'')
    
    if name in CLASS_MAPPINGS:
        return CLASS_MAPPINGS[name]
    
    for valid_class in VALID_CLASSES:
        if re.search(r'\b' + re.escape(valid_class.lower()) + r'\b', name.lower()):
            return valid_class
            
    return None

def extract_class_from_yolo_txt(txt_path, class_names):
    detections = []
    try:
        with open(txt_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if class_id < len(class_names):
                        raw_class = class_names[class_id]
                        normalized_class = normalize_class_name(raw_class)
                        if normalized_class:
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            bbox = (center_x, center_y, width, height)
                            detections.append((normalized_class, bbox))
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
    return detections

def yolo_to_pixels(bbox, img_width, img_height):
    center_x, center_y, w, h = bbox
    x_center_px = center_x * img_width
    y_center_px = center_y * img_height
    width_px = w * img_width
    height_px = h * img_height

    left = int(x_center_px - (width_px / 2))
    upper = int(y_center_px - (height_px / 2))
    right = int(x_center_px + (width_px / 2))
    lower = int(y_center_px + (height_px / 2))

    left = max(0, left)
    upper = max(0, upper)
    right = min(img_width, right)
    lower = min(img_height, lower)

    if right <= left:
        right = left + 1
    if lower <= upper:
        lower = upper + 1

    return left, upper, right, lower

def extract_class_from_filename(filename, class_names=None):
    filename_lower = filename.lower()
    name_parts = re.split(r'[-_\s.]', filename_lower)
    
    for part in name_parts:
        part = part.strip()
        if not part:
            continue
            
        for class_name in VALID_CLASSES:
            if class_name.lower() == part:
                return [class_name]
        
        for key, val in CLASS_MAPPINGS.items():
            if val and key.lower() == part:
                return [val]
    
    special_cases = {
        'straumann': 'ITI',
        'biocare': 'nobel'
    }
    
    for special, class_name in special_cases.items():
        if special in filename_lower:
            return [class_name]
    
    for class_name in VALID_CLASSES:
        class_lower = class_name.lower()
        if re.search(r'\b' + re.escape(class_lower) + r'\b', filename_lower):
            return [class_name]
    
    return []

def convert_dataset(dataset_path, output_base_path, cropped_output_base_path, create_val=True, val_split=0.2):
    dataset_name = os.path.basename(dataset_path)
    output_path = os.path.join(output_base_path, dataset_name)
    cropped_output_path = os.path.join(cropped_output_base_path, dataset_name)

    print(f"\n=== Processing dataset: {dataset_name} ===")

    yaml_path = os.path.join(dataset_path, 'data.yaml')
    raw_class_names = []
    if os.path.exists(yaml_path):
        try:
            raw_class_names = read_yaml_classes(yaml_path)
            if isinstance(raw_class_names, list):
                 print(f"Found YAML class definitions: {raw_class_names}")
            else:
                 print(f"Warning: 'names' in {yaml_path} is not a list. Found: {type(raw_class_names)}. Will rely on filename detection.")
                 raw_class_names = []
        except Exception as e:
            print(f"Error reading or parsing {yaml_path}: {e}. Will rely on filename detection.")
            raw_class_names = []
    else:
        print(f"Warning: No data.yaml found in {dataset_path}, will rely on filename detection only")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(cropped_output_path, exist_ok=True)
    for base in [output_path, cropped_output_path]:
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(base, split)
            os.makedirs(split_dir, exist_ok=True)
            for class_name in VALID_CLASSES:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    has_val = os.path.exists(os.path.join(dataset_path, 'val', 'images'))

    image_data = {'train': [], 'val': [], 'test': []}

    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(dataset_path, split, 'images')
        if not os.path.exists(split_images_dir):
            print(f"Split {split} not found in {dataset_name}")
            continue

        print(f"Processing {split} split...")

        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
            image_files.extend(glob.glob(os.path.join(split_images_dir, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(split_images_dir, f'*.{ext.upper()}')))
        image_files = sorted(list(set(image_files)))

        print(f"Found {len(image_files)} images in {split} split")

        for img_path in image_files:
            img_filename = os.path.basename(img_path)
            img_name, img_ext = os.path.splitext(img_filename)

            label_path = os.path.join(dataset_path, split, 'labels', f"{img_name}.txt")
            classes_for_full_image = set()
            detections = []

            if os.path.exists(label_path) and raw_class_names:
                raw_detections = extract_class_from_yolo_txt(label_path, raw_class_names)
                valid_detections = [(cls, bbox) for cls, bbox in raw_detections if cls in VALID_CLASSES]
                if valid_detections:
                    detections.extend(valid_detections)
                    classes_for_full_image.update(cls for cls, _ in valid_detections)

            if not classes_for_full_image:
                filename_classes = extract_class_from_filename(img_name)
                valid_filename_classes = [cls for cls in filename_classes if cls in VALID_CLASSES]
                if valid_filename_classes:
                    classes_for_full_image.update(valid_filename_classes)

            if classes_for_full_image:
                 image_data[split].append({
                     'path': img_path,
                     'full_classes': list(classes_for_full_image),
                     'detections': detections
                 })

    final_image_assignments = {'train': [], 'val': [], 'test': []}
    train_img_paths_after_split = set()
    val_img_paths_after_split = set()

    if create_val and not has_val and image_data['train']:
        print(f"Creating validation set from training data (split: {val_split})")

        images_by_class = {}
        for data in image_data['train']:
            if data['full_classes']:
                cls_for_split = data['full_classes'][0]
                if cls_for_split not in images_by_class:
                    images_by_class[cls_for_split] = []
                images_by_class[cls_for_split].append(data)

        temp_train_images = []
        temp_val_images = []

        for cls, imgs_in_class in images_by_class.items():
            random.shuffle(imgs_in_class)
            num_imgs = len(imgs_in_class)
            if num_imgs < 2:
                temp_train_images.extend(imgs_in_class)
                continue
            val_size = max(1, int(num_imgs * val_split))
            if val_size >= num_imgs: val_size = num_imgs - 1

            temp_val_images.extend(imgs_in_class[:val_size])
            temp_train_images.extend(imgs_in_class[val_size:])

        final_image_assignments['train'] = temp_train_images
        final_image_assignments['val'] = temp_val_images
        final_image_assignments['test'] = image_data['test']

        train_img_paths_after_split = {d['path'] for d in temp_train_images}
        val_img_paths_after_split = {d['path'] for d in temp_val_images}

        print(f"Split complete: {len(temp_train_images)} train, {len(temp_val_images)} val")

    else:
        final_image_assignments = copy.deepcopy(image_data)
        train_img_paths_after_split = {d['path'] for d in final_image_assignments['train']}
        if has_val:
             val_img_paths_after_split = {d['path'] for d in final_image_assignments['val']}

    class_counter = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    cropped_class_counter = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    copied_full_count = 0
    skipped_full_count = 0
    total_cropped_count = 0
    total_skipped_crops = 0
    unique_copied_full_paths = set()

    print("\nProcessing images for saving...")

    for original_split in ['train', 'val', 'test']:
        for data in final_image_assignments[original_split]:
            img_path = data['path']
            img_filename = os.path.basename(img_path)
            img_name, img_ext = os.path.splitext(img_filename)
            full_classes = data['full_classes']
            detections = data['detections']

            final_split = original_split
            if create_val and not has_val:
                if img_path in val_img_paths_after_split:
                    final_split = 'val'
                elif img_path in train_img_paths_after_split:
                    final_split = 'train'

            copied_this_image_full = False
            for class_name in full_classes:
                dest_full_path = os.path.join(output_path, final_split, class_name, img_filename)
                try:
                    if dest_full_path not in unique_copied_full_paths:
                        shutil.copy(img_path, dest_full_path)
                        unique_copied_full_paths.add(dest_full_path)
                        copied_this_image_full = True
                    class_counter[final_split][class_name] += 1
                except Exception as e:
                    print(f"Error copying {img_path} to {dest_full_path}: {e}")
                    skipped_full_count += 1

            if detections:
                try:
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_width, img_height = img.size
                        crop_index = 0
                        for class_name, yolo_bbox in detections:
                            pixel_bbox = yolo_to_pixels(yolo_bbox, img_width, img_height)
                            if pixel_bbox[2] > pixel_bbox[0] and pixel_bbox[3] > pixel_bbox[1]:
                                try:
                                    cropped_img = img.crop(pixel_bbox)
                                    cropped_filename = f"{img_name}_crop{crop_index}.jpg"
                                    dest_crop_path = os.path.join(cropped_output_path, final_split, class_name, cropped_filename)
                                    cropped_img.save(dest_crop_path, quality=95)
                                    cropped_class_counter[final_split][class_name] += 1
                                    total_cropped_count += 1
                                    crop_index += 1
                                except Exception as crop_err:
                                     print(f"Error cropping/saving {img_filename} (crop {crop_index}), class {class_name}: {crop_err}")
                                     total_skipped_crops += 1
                            else:
                                total_skipped_crops += 1
                except Exception as e:
                    print(f"Error opening/processing image {img_path} for cropping: {e}")
                    total_skipped_crops += len(detections)

    copied_full_count = len(unique_copied_full_paths)

    print(f"\nFinished processing {dataset_name}:")
    print(f"  - Full Images: Copied {copied_full_count} files, Skipped/Errors {skipped_full_count} references.")
    print(f"  - Cropped Images: Generated {total_cropped_count} files, Skipped/Errors {total_skipped_crops} crops.")

    print("\nFinal Full Image Distribution:")
    for split in ['train', 'val', 'test']:
        if class_counter[split]: print(f"  {split}: {dict(class_counter[split])}")
        else: print(f"  {split}: No images.")

    print("\nFinal Cropped Image Distribution:")
    for split in ['train', 'val', 'test']:
        if cropped_class_counter[split]: print(f"  {split}: {dict(cropped_class_counter[split])}")
        else: print(f"  {split}: No images.")

    if copied_full_count == 0 and total_cropped_count == 0:
        print(f"\nWARNING: No images were copied or cropped for dataset {dataset_name}. Check class names/filenames/annotations.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_collected_dir = os.path.join(project_root, 'data', 'data_collected')
    output_base_dir = os.path.join(project_root, 'data', 'classification_datasets')
    cropped_output_base_dir = os.path.join(project_root, 'data', 'cropped_datasets')

    if not os.path.exists(data_collected_dir):
        print(f"Error: Directory not found: {data_collected_dir}")
        sys.exit(1)

    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(cropped_output_base_dir, exist_ok=True)
    random.seed(42)

    datasets = [d for d in os.listdir(data_collected_dir)
                if os.path.isdir(os.path.join(data_collected_dir, d))]
    print(f"Found {len(datasets)} datasets: {datasets}")

    for dataset in datasets:
        dataset_path = os.path.join(data_collected_dir, dataset)
        convert_dataset(dataset_path, output_base_dir, cropped_output_base_dir, create_val=True, val_split=0.2)

    print("\nConversion complete!")

if __name__ == "__main__":
    main()