import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
)

class DentalImplantEfficientNetB3:
    """
    EfficientNetB3-based model for dental implant classification in radiographic images.
    Implements transfer learning with progressive unfreezing for optimal performance.
    """
    
    def __init__(self, input_shape=(512, 512, 3), num_classes=None, weights='imagenet'):
        """
        Initialize the EfficientNetB3 model for dental implant classification.
        
        Args:
            input_shape (tuple): The input image dimensions (height, width, channels)
            num_classes (int): Number of implant classes to classify
            weights (str): Pre-trained weights to use ('imagenet' or None)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.model = None
        
    def build(self):
        """
        Build the EfficientNetB3 model with a custom classification head.
        
        Returns:
            tensorflow.keras.Model: Compiled EfficientNetB3 model
        """
        if not self.num_classes:
            raise ValueError("Number of classes must be specified")
            
        # Create input layer
        inputs = Input(shape=self.input_shape)
        
        # Base model with pre-trained weights
        base_model = EfficientNetB3(
            include_top=False,
            weights=self.weights,
            input_tensor=inputs,
            input_shape=self.input_shape
        )
        
        # Freeze base model for initial training
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Combine base model and custom head
        self.model = Model(inputs=inputs, outputs=outputs)
        
        return self.model
    
    def get_base_model_layers(self):
        """
        Get the layers of the base model for progressive unfreezing.
        
        Returns:
            list: List of base model layers
        """
        # Find the base model layers (excluding the classification head)
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.Model):  # This is the base model
                return layer.layers
        return []
    
    def unfreeze_top_layers(self, percentage=0.3):
        """
        Unfreeze the top percentage of the base model layers for fine-tuning.
        
        Args:
            percentage (float): Percentage of layers to unfreeze from the top
        """
        base_layers = self.get_base_model_layers()
        
        if not base_layers:
            return
            
        # Calculate the number of layers to unfreeze
        num_to_unfreeze = int(len(base_layers) * percentage)
        
        # Freeze/unfreeze layers accordingly
        for layer in base_layers:
            layer.trainable = False
            
        for layer in base_layers[-num_to_unfreeze:]:
            layer.trainable = True
    
    def unfreeze_all_layers(self):
        """
        Unfreeze all layers in the model for final fine-tuning.
        """
        for layer in self.get_base_model_layers():
            layer.trainable = True

    def freeze_base_model(self):
        """
        Freeze all layers in the base model for the initial training phase.
        Only the top classification layers will be trainable.
        """
        # First check if model is already built
        if not hasattr(self, 'model') or self.model is None:
            self.build()
        
        # In an EfficientNetB3 model created with transfer learning,
        # we need to freeze all layers except our custom top layers
        # The standard pattern is:
        # 1. Get the base model structure
        # 2. Freeze all layers in it
        
        # Find the base EfficientNetB3 layers - typically it's the first functional layer in the model
        base_model = None
        for layer in self.model.layers:
            if layer.name.startswith('efficientnet'):
                base_model = layer
                break
        
        if base_model:
            # Freeze all layers in the base model
            for layer in base_model.layers:
                layer.trainable = False
            print(f"Froze all {len(base_model.layers)} layers in EfficientNet base model")
        else:
            # Alternative approach if we can't find the base model:
            # In a standard EfficientNetB3 setup, the classification layers are added after 
            # the base model, so we freeze all layers except the last few
            trainable_layers = 3  # Typically the final classification layers
            for layer in self.model.layers[:-trainable_layers]:
                layer.trainable = False
            print(f"Froze all layers except the last {trainable_layers} layers")
        
        print("Base model layers frozen")