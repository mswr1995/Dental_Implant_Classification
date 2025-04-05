#!/usr/bin/env python3
"""
Dental Implant Classification - Debug Script

This script provides a comprehensive environment for testing different combinations of:
1. Data sources
2. Image processing techniques
3. Model architectures

Usage:
    python dental_implant_debug.py [--source SOURCE] [--process PROCESS] [--model MODEL] 
                                  [--lr LEARNING_RATE] [--batch BATCH_SIZE] [--epochs EPOCHS]
                                  [--mode {interactive,auto}] [--compare] [--cpu]

The goal is to identify the optimal approach for dental implant classification.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import cv2
from skimage import io, color, exposure, filters, util

# Configure TensorFlow to manage GPU memory and suppress warnings
def configure_tensorflow(use_gpu=True):
    """Configure TensorFlow to handle GPU memory and suppress warnings"""
    # Disable verbose logging
    tf.get_logger().setLevel('ERROR')
    
    # Only display errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    if not use_gpu:
        print("Forcing CPU usage (GPU disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    
    # Prevent TensorFlow from pre-allocating GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"GPU memory growth enabled on {len(physical_devices)} GPU(s)")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Get the absolute path of the script
script_path = os.path.dirname(os.path.abspath(__file__))

# Set base paths
BASE_PATH = script_path
DATA_PATH = os.path.join(BASE_PATH, 'data')
RESULTS_PATH = os.path.join(BASE_PATH, 'results')

# Create results directory structure if it doesn't exist
os.makedirs(os.path.join(RESULTS_PATH, 'logs'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'models'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'plots'), exist_ok=True)

# Configuration settings dictionary
config = {
    'data_source': None,  # Will be set based on arguments or interactive selection
    'processing_method': 'original',  # Options: 'original', 'denoised', 'enhanced', 'sharpened'
    'model_type': 'efficientnetb3',  # Options: 'efficientnetb3', 'custom_cnn', 'resnet50', 'densenet121'
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 10,
    'img_channels': 3,  # Will be set based on processing method
    'input_shape': None,  # Will be determined from data
    'num_classes': None,  # Will be determined from data
    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'use_gpu': True,  # Whether to use GPU or force CPU
}

# Parse command line arguments
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dental Implant Classification Debug Script')
    
    # Add arguments
    parser.add_argument('--source', type=str, help='Data source name (must be in the data directory)')
    parser.add_argument('--source-idx', type=int, help='Data source index (1-based, use --list-sources to see indexes)')
    parser.add_argument('--process', type=str, choices=['original', 'denoised', 'enhanced', 'sharpened'], 
                        default='original', help='Image processing method')
    parser.add_argument('--model', type=str, choices=['efficientnetb3', 'custom_cnn', 'resnet50', 'densenet121'], 
                        default='efficientnetb3', help='Model architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--mode', type=str, choices=['interactive', 'auto'], default='interactive',
                        help='Run mode: interactive prompts or automatic with arguments')
    parser.add_argument('--compare', action='store_true', help='Compare experiments after training')
    parser.add_argument('--visualize', action='store_true', help='Visualize model attention after training')
    parser.add_argument('--list-sources', action='store_true', help='List available data sources and exit')
    parser.add_argument('--list-experiments', action='store_true', help='List available experiments and exit')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage instead of GPU')
    
    return parser.parse_args()

# =============== DATA SOURCE FUNCTIONS ===============

def scan_data_sources():
    """Scan for available data sources in the data directory"""
    sources = []
    # Look for directories in the data folder
    for item in os.listdir(DATA_PATH):
        source_path = os.path.join(DATA_PATH, item)
        if os.path.isdir(source_path):
            # Check if it contains train/val/test subdirectories
            if all(os.path.isdir(os.path.join(source_path, split)) for split in ['train', 'val', 'test']):
                sources.append(item)

    return sources

def get_class_distribution(data_dir):
    """Get number of images per class"""
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            class_counts[class_name] = count

    return class_counts

def display_source_info(source):
    """Display information about the selected data source"""
    source_path = os.path.join(DATA_PATH, source)

    # Get class distribution for each split
    splits = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(source_path, split)
        class_counts = get_class_distribution(split_path)
        splits[split] = class_counts

    # Calculate total images
    total_images = sum(sum(counts.values()) for counts in splits.values())

    # Display summary
    print(f"=== Data Source: {source} ===")
    print(f"Total images: {total_images}")
    # Display class distribution
    for split, counts in splits.items():
        print(f"\n{split.capitalize()} set:")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count} images")

    # Get image sizes
    train_path = os.path.join(source_path, 'train')
    class_dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    if class_dirs:
        first_class = class_dirs[0]
        class_path = os.path.join(train_path, first_class)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            sample_img_path = os.path.join(class_path, image_files[0])
            img = plt.imread(sample_img_path)
            print(f"\nImage dimensions: {img.shape}")
            # Set the input shape based on the first image
            if len(img.shape) == 2:  # Grayscale
                config['input_shape'] = (img.shape[0], img.shape[1], 1)
                config['img_channels'] = 1
            else:  # RGB
                config['input_shape'] = img.shape
                config['img_channels'] = img.shape[2]

    # Set number of classes
    config['num_classes'] = len(splits['train'])

    return splits

def display_sample_images(source, num_samples=3, save_figure=True):
    """Display sample images from each class in the selected source"""
    train_path = os.path.join(DATA_PATH, source, 'train')
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 3*num_classes))

    # Handle the case where there's only one class
    if num_classes == 1:
        axes = [axes]

    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Select random samples
        import random
        samples = random.sample(image_files, min(num_samples, len(image_files)))

        for j, file in enumerate(samples):
            img_path = os.path.join(class_path, file)
            img = plt.imread(img_path)

            # Handle grayscale vs RGB display
            if len(img.shape) == 2:
                axes[i][j].imshow(img, cmap='gray')
            else:
                axes[i][j].imshow(img)

            axes[i][j].set_title(f"{class_name}\n{file}")
            axes[i][j].axis('off')

    plt.tight_layout()
    
    if save_figure:
        # Save the sample images figure
        samples_path = os.path.join(RESULTS_PATH, 'plots', f"{source}_samples.png")
        plt.savefig(samples_path)
        print(f"Sample images saved to: {samples_path}")
    
    plt.show()

# =============== IMAGE PROCESSING FUNCTIONS ===============

def apply_processing(img, method='original'):
    """Apply selected processing method to an image"""
    # Convert to float for processing if not already
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    if method == 'original':
        # No processing
        processed = img

    elif method == 'denoised':
        # Apply denoising
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            processed = cv2.fastNlMeansDenoisingColored(
                (img * 255).astype(np.uint8), None, 10, 10, 7, 21)
            processed = processed.astype(np.float32) / 255.0
        else:  # Grayscale
            if len(img.shape) == 3:  # Extra dimension
                img_gray = img[:,:,0]
            else:
                img_gray = img
            processed = cv2.fastNlMeansDenoising(
                (img_gray * 255).astype(np.uint8), None, 10, 7, 21)
            processed = processed.astype(np.float32) / 255.0
            # Restore shape if needed
            if len(img.shape) == 3 and processed.ndim == 2:
                processed = np.expand_dims(processed, axis=2)

    elif method == 'enhanced':
        # Apply contrast enhancement
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            processed_lab = cv2.merge((cl, a, b))
            processed = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB)
            processed = processed.astype(np.float32) / 255.0
        else:  # Grayscale
            if len(img.shape) == 3:  # Extra dimension
                img_gray = img[:,:,0]
            else:
                img_gray = img
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply((img_gray * 255).astype(np.uint8))
            processed = processed.astype(np.float32) / 255.0
            # Restore shape if needed
            if len(img.shape) == 3 and processed.ndim == 2:
                processed = np.expand_dims(processed, axis=2)

    elif method == 'sharpened':
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            processed = cv2.filter2D((img * 255).astype(np.uint8), -1, kernel)
            processed = processed.astype(np.float32) / 255.0
        else:  # Grayscale
            if len(img.shape) == 3:  # Extra dimension
                img_gray = img[:,:,0]
            else:
                img_gray = img
            processed = cv2.filter2D((img_gray * 255).astype(np.uint8), -1, kernel)
            processed = processed.astype(np.float32) / 255.0
            # Restore shape if needed
            if len(img.shape) == 3 and processed.ndim == 2:
                processed = np.expand_dims(processed, axis=2)

    # Ensure values are in [0, 1] range
    processed = np.clip(processed, 0, 1)

    return processed

def display_processing_comparison(source, method, save_figure=True):
    """Display comparison of original and processed images"""
    if not source:
        print("Please set a data source first.")
        return

    train_path = os.path.join(DATA_PATH, source, 'train')
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    # Select one image from each class
    fig, axes = plt.subplots(len(classes), 2, figsize=(10, 4*len(classes)))

    # Handle case with only one class
    if len(classes) == 1:
        axes = [axes]

    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if image_files:
            # Select first image
            img_path = os.path.join(class_path, image_files[0])
            img = plt.imread(img_path)

            # Convert to float for processing if not already
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0

            # Apply processing
            processed_img = apply_processing(img, method)

            # Display original
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                axes[i][0].imshow(img, cmap='gray')
            else:
                axes[i][0].imshow(img)
            axes[i][0].set_title(f"{class_name} - Original")
            axes[i][0].axis('off')

            # Display processed
            if len(processed_img.shape) == 2 or (len(processed_img.shape) == 3 and processed_img.shape[2] == 1):
                axes[i][1].imshow(processed_img, cmap='gray')
            else:
                axes[i][1].imshow(processed_img)
            axes[i][1].set_title(f"{class_name} - {method.capitalize()}")
            axes[i][1].axis('off')

    plt.tight_layout()
    
    if save_figure:
        # Save the processing comparison figure
        comparison_path = os.path.join(RESULTS_PATH, 'plots', f"{source}_{method}_comparison.png")
        plt.savefig(comparison_path)
        print(f"Processing comparison saved to: {comparison_path}")
    
    plt.show()

def list_experiments():
    """List available experiments in the results directory"""
    experiments_path = os.path.join(RESULTS_PATH, 'metrics')
    experiments = [exp for exp in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, exp))]
    return experiments

# =============== MODEL BUILDING FUNCTIONS ===============

def create_model(model_type, input_shape, num_classes):
    """Create and compile a model based on the specified type"""
    print(f"Creating {model_type} model for input shape {input_shape} with {num_classes} classes")
    
    if model_type == 'efficientnetb3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
    elif model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
    elif model_type == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
    elif model_type == 'custom_cnn':
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators(source, processing_method):
    """Create training, validation, and test data generators"""
    source_path = os.path.join(DATA_PATH, source)
    
    # Image data generators
    if processing_method == 'original':
        # Standard augmentation only
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation and test sets
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
    else:
        # For custom processing methods, use preprocessing function
        def preprocess_input(img):
            return apply_processing(img, method=processing_method)
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(source_path, 'train'),
        target_size=(config['input_shape'][0], config['input_shape'][1]),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(source_path, 'val'),
        target_size=(config['input_shape'][0], config['input_shape'][1]),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(source_path, 'test'),
        target_size=(config['input_shape'][0], config['input_shape'][1]),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def train_and_evaluate_model(model, train_generator, val_generator, test_generator):
    """Train and evaluate the model"""
    experiment_name = f"{config['data_source']}_{config['processing_method']}_{config['model_type']}_{config['timestamp']}"
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(RESULTS_PATH, 'models', f"{experiment_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(RESULTS_PATH, 'logs', experiment_name)
        )
    ]
    
    # Train the model
    print(f"\nTraining model: {config['model_type']}")
    print(f"Processing method: {config['processing_method']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    
    history = model.fit(
        train_generator,
        epochs=config['epochs'],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    test_results = model.evaluate(test_generator, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    
    # Save evaluation metrics
    metrics_dir = os.path.join(RESULTS_PATH, 'metrics', experiment_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save history and test results
    results = {
        'config': config,
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'history': {k: [float(v) for v in history.history[k]] for k in history.history.keys()}
    }
    
    with open(os.path.join(metrics_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate and save plots
    generate_plots(history, experiment_name)
    
    # Generate classification report and confusion matrix
    generate_classification_report(model, test_generator, experiment_name)
    
    return history, test_results

def generate_plots(history, experiment_name):
    """Generate training history plots"""
    plots_dir = os.path.join(RESULTS_PATH, 'plots')
    
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{experiment_name}_training.png"))
    plt.close()

def generate_classification_report(model, test_generator, experiment_name):
    """Generate and save classification report and confusion matrix"""
    metrics_dir = os.path.join(RESULTS_PATH, 'metrics', experiment_name)
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Class names
    class_names = list(test_generator.class_indices.keys())
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save classification report
    with open(os.path.join(metrics_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    """Main function to run the script"""
    print("=" * 80)
    print("Dental Implant Classification - Debug Script")
    print("=" * 80)
    
    # Parse command line arguments
    args = parse_args()
    
    # Set GPU usage based on arguments
    if args.cpu:
        config['use_gpu'] = False
    
    # Configure TensorFlow with GPU/CPU selection
    configure_tensorflow(use_gpu=config['use_gpu'])
    
    # List available data sources if requested
    sources = scan_data_sources()
    if not sources:
        print("No data sources found in", DATA_PATH)
        print("Please make sure your data is organized with train/val/test subdirectories")
        return
    
    # Just list sources and exit if requested
    if args.list_sources:
        print("\nAvailable data sources:")
        for i, source in enumerate(sources):
            print(f"{i+1}. {source}")
        return
    
    # Just list experiments and exit if requested
    if args.list_experiments:
        available_experiments = list_experiments()
        if available_experiments:
            print("\nAvailable experiments:")
            for i, exp in enumerate(available_experiments):
                print(f"{i+1}. {exp}")
        else:
            print("No experiments found.")
        return
    
    # Set mode based on arguments
    mode = args.mode
    
    # If in automatic mode and source index is specified, validate and set it
    if mode == 'auto' and args.source_idx is not None:
        if 1 <= args.source_idx <= len(sources):
            config['data_source'] = sources[args.source_idx - 1]  # Convert to 0-based index
        else:
            print(f"Error: Source index {args.source_idx} is out of range. Available sources (1-{len(sources)}):")
            for i, source in enumerate(sources):
                print(f"  {i+1}. {source}")
            return
    # If source name is specified directly, use that instead
    elif mode == 'auto' and args.source:
        if args.source in sources:
            config['data_source'] = args.source
        else:
            print(f"Error: Source '{args.source}' not found. Available sources:")
            for i, source in enumerate(sources):
                print(f"  {i+1}. {source}")
            return
    
    # In interactive mode or if no source specified in auto mode
    if mode == 'interactive' or (mode == 'auto' and not config['data_source']):
        print("\nAvailable data sources:")
        for i, source in enumerate(sources):
            print(f"{i+1}. {source}")
        
        # Select data source
        while True:
            try:
                source_idx = int(input("\nSelect a data source (number): ")) - 1
                if 0 <= source_idx < len(sources):
                    config['data_source'] = sources[source_idx]
                    break
                else:
                    print("Invalid selection. Please enter a number from the list.")
            except ValueError:
                print("Please enter a valid number.")
    
    print(f"\nSelected data source: {config['data_source']}")
    display_source_info(config['data_source'])
    
    # Set up other configuration if in interactive mode
    if mode == 'interactive':
        # Select processing method
        print("\nSelect image processing method:")
        print("1. Original (no processing)")
        print("2. Denoised (noise reduction)")
        print("3. Enhanced (contrast enhancement)")
        print("4. Sharpened (edge enhancement)")
        
        while True:
            try:
                process_choice = int(input("Enter your choice (1-4): "))
                if 1 <= process_choice <= 4:
                    method_map = {1: 'original', 2: 'denoised', 3: 'enhanced', 4: 'sharpened'}
                    config['processing_method'] = method_map[process_choice]
                    break
                else:
                    print("Invalid selection. Please enter a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Show processing example
        display_processing_comparison(config['data_source'], config['processing_method'])
        
        # Select model type
        print("\nSelect model architecture:")
        print("1. EfficientNetB3 (transfer learning)")
        print("2. Custom CNN (from scratch)")
        print("3. ResNet50 (transfer learning)")
        print("4. DenseNet121 (transfer learning)")
        
        while True:
            try:
                model_choice = int(input("Enter your choice (1-4): "))
                if 1 <= model_choice <= 4:
                    model_map = {1: 'efficientnetb3', 2: 'custom_cnn', 3: 'resnet50', 4: 'densenet121'}
                    config['model_type'] = model_map[model_choice]
                    break
                else:
                    print("Invalid selection. Please enter a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Set hyperparameters
        try:
            config['learning_rate'] = float(input("\nEnter learning rate (default=0.001): ") or "0.001")
            config['batch_size'] = int(input("Enter batch size (default=16): ") or "16")
            config['epochs'] = int(input("Enter number of epochs (default=10): ") or "10")
        except ValueError:
            print("Invalid input, using default values.")
        
        # Add CPU/GPU selection in interactive mode
        if not args.cpu:  # Only ask if not already set via command line
            use_gpu_response = input("\nUse GPU for training? (y/n, default=y): ").lower()
            if use_gpu_response.startswith('n'):
                config['use_gpu'] = False
                # Reconfigure TensorFlow if user chose CPU in interactive mode
                configure_tensorflow(use_gpu=False)
                print("Switched to CPU-only mode")
    else:
        # In auto mode, use command line args
        config['processing_method'] = args.process
        config['model_type'] = args.model
        config['learning_rate'] = args.lr
        config['batch_size'] = args.batch
        config['epochs'] = args.epochs
    
    # Create data generators
    print("\nPreparing data generators...")
    train_generator, val_generator, test_generator = create_data_generators(
        config['data_source'], config['processing_method']
    )
    
    # Create model
    model = create_model(
        config['model_type'], config['input_shape'], config['num_classes']
    )
    model.summary()
    
    # Add resource warning for large models
    if not config['use_gpu'] and config['model_type'] == 'custom_cnn':
        print("\nWarning: Training a custom CNN on CPU may be slow, especially with large images.")
        print("If you encounter memory issues, consider using a smaller batch size.")
    
    # Train and evaluate model
    try:
        history, test_results = train_and_evaluate_model(
            model, train_generator, val_generator, test_generator
        )
        
        print("\nExperiment complete!")
        print(f"Test accuracy: {test_results[1]:.4f}")
        
        # Save final model
        experiment_name = f"{config['data_source']}_{config['processing_method']}_{config['model_type']}_{config['timestamp']}"
        model.save(os.path.join(RESULTS_PATH, 'models', f"{experiment_name}_final.h5"))
        print(f"Model saved as: {experiment_name}_final.h5")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        if "out of memory" in str(e).lower() or "resource exhausted" in str(e).lower():
            print("\nMemory error detected. Troubleshooting tips:")
            print("1. Try using CPU instead of GPU: --cpu")
            print("2. Try using a smaller batch size: --batch 4")
            print("3. Try using a different model architecture")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
