import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)

class DentalImplantCustomCNN:
    """
    Custom CNN model for dental implant classification in radiographic images.
    """
    
    def __init__(self, input_shape=(512, 512, 3), num_classes=None):
        """
        Initialize the Custom CNN model for dental implant classification.
        
        Args:
            input_shape (tuple): The input image dimensions (height, width, channels)
            num_classes (int): Number of implant classes to classify
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build(self):
        """
        Build the Custom CNN model architecture.
        
        Returns:
            tensorflow.keras.Model: Compiled Custom CNN model
        """
        if not self.num_classes:
            raise ValueError("Number of classes must be specified")
            
        # Define the Sequential model
        model = Sequential([
            # First convolution block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Second convolution block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Third convolution block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Fourth convolution block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Fifth convolution block
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling2D(),

            # Classification head
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return self.model
    
    def get_base_model_layers(self):
        """
        Get the layers of the base model for progressive unfreezing.
        Not needed for custom CNN, but included for API consistency.
        
        Returns:
            list: List of model layers
        """
        return self.model.layers
    
    def unfreeze_top_layers(self, percentage=0.3):
        """
        Unfreeze the top percentage of layers.
        Not needed for custom CNN as we train all layers, but included for API consistency.
        
        Args:
            percentage (float): Percentage of layers to unfreeze from the top
        """
        # Not implemented for custom CNN as we train all layers from the beginning
        pass
    
    def unfreeze_all_layers(self):
        """
        Unfreeze all layers.
        Not needed for custom CNN as we train all layers, but included for API consistency.
        """
        # All layers are already trainable in custom CNN
        pass