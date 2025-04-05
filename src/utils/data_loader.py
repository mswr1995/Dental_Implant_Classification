import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DentalImplantDataLoader:
    """
    Data loader for dental implant radiographic images with augmentation.
    """
    
    def __init__(self, img_size=(512, 512), batch_size=16):
        """
        Initialize the data loader with specified parameters.
        
        Args:
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.class_weights = None
        self.class_indices = None
        self.num_classes = None
        
    def create_train_generator(self, data_dir, augmentation=True):
        """
        Create training data generator with augmentation.
        
        Args:
            data_dir (str): Path to training data directory
            augmentation (bool): Whether to apply data augmentation
            
        Returns:
            DirectoryIterator: Training data generator
        """
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
                fill_mode='constant',
                cval=0,
                validation_split=0.0  # No validation split here, we use separate directories
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.class_indices = self.train_generator.class_indices
        self.num_classes = len(self.class_indices)
        
        return self.train_generator
    
    def create_val_generator(self, data_dir):
        """
        Create validation data generator without augmentation.
        
        Args:
            data_dir (str): Path to validation data directory
            
        Returns:
            DirectoryIterator: Validation data generator
        """
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        self.val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return self.val_generator
    
    def create_test_generator(self, data_dir):
        """
        Create test data generator without augmentation.
        
        Args:
            data_dir (str): Path to test data directory
            
        Returns:
            DirectoryIterator: Test data generator
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return self.test_generator
    
    def calculate_class_weights(self):
        """
        Calculate class weights based on the distribution of samples.
        
        Returns:
            dict: Class weights dictionary
        """
        if not self.train_generator:
            raise ValueError("Training generator must be created first")
            
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Get class distribution
        class_counts = self.train_generator.classes
        unique_classes = np.unique(class_counts)
        
        # Calculate class weights
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=class_counts
        )
        
        self.class_weights = {i: weights[i] for i in range(len(weights))}
        return self.class_weights