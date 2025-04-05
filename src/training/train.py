import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
import datetime
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ProgressiveTrainer:
    """
    Implementation of progressive training strategy for dental implant classification.
    """
    
    def __init__(
        self,
        train_data_dir,
        val_data_dir,
        model_save_dir,
        img_size=(512, 512),
        batch_size=16,
        use_class_weights=True,
        model_type='efficientnet'  # Added model_type parameter
    ):
        """
        Initialize the trainer with specified parameters.
        
        Args:
            train_data_dir (str): Path to training data directory
            val_data_dir (str): Path to validation data directory
            model_save_dir (str): Path to save model checkpoints
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for training
            use_class_weights (bool): Whether to use class weights for imbalanced classes
            model_type (str): Type of model to train ('efficientnet' or 'custom_cnn')
        """
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.model_save_dir = model_save_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.use_class_weights = use_class_weights
        self.model_type = model_type
        
        # Create directories if they don't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Import here to avoid circular imports
        from src.utils.data_loader import DentalImplantDataLoader
        
        # Initialize data loader
        self.data_loader = DentalImplantDataLoader(img_size=img_size, batch_size=batch_size)
        
        # Initialize model
        self.model = None
        self.history = None
        self.model_name = 'model'  # Will be set properly in create_model
        
        # Add a timestamp to uniquely identify this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_metadata = {
            "model_type": model_type,
            "img_size": img_size,
            "batch_size": batch_size,
            "use_class_weights": use_class_weights,
            "timestamp": self.timestamp,
            "stages": {}
        }
        
    def prepare_data(self):
        """
        Prepare training and validation data generators.
        """
        # Create training data generator with augmentation
        self.train_generator = self.data_loader.create_train_generator(
            self.train_data_dir, augmentation=True
        )
        
        # Create validation data generator
        self.val_generator = self.data_loader.create_val_generator(self.val_data_dir)
        
        # Calculate class weights if needed
        if self.use_class_weights:
            self.class_weights = self.data_loader.calculate_class_weights()
        else:
            self.class_weights = None
            
        print(f"Found {self.train_generator.samples} training samples in {self.data_loader.num_classes} classes")
        print(f"Found {self.val_generator.samples} validation samples")
        
    def create_model(self):
        """
        Create and compile the model based on model_type.
        
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        if self.model_type == 'efficientnet':
            from src.models.efficientnet_model import DentalImplantEfficientNetB3
            
            # Initialize model
            model_class = DentalImplantEfficientNetB3(
                input_shape=(self.img_size[0], self.img_size[1], 3),
                num_classes=self.data_loader.num_classes
            )
            self.model_name = 'efficientnetb3'
        else:  # custom_cnn
            from src.models.custom_cnn import DentalImplantCustomCNN
            
            # Initialize model
            model_class = DentalImplantCustomCNN(
                input_shape=(self.img_size[0], self.img_size[1], 3),
                num_classes=self.data_loader.num_classes
            )
            self.model_name = 'custom_cnn'
        
        # Build model
        self.model = model_class.build()
        
        # Store model class for later use (unfreezing etc.)
        self.model_class = model_class
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def _save_model(self, stage, is_best=False):
        """
        Save model with consistent naming and metadata.
        
        Args:
            stage (int): Training stage (1, 2, or 3)
            is_best (bool): Whether this is the best model for this stage
        """
        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Define model type (best or final)
        model_type = "best" if is_best else "final"
        
        # Create filename with timestamp to prevent overwriting
        filename = f"{self.model_name}_stage_{stage}_{model_type}_{self.timestamp}.keras"
        filepath = os.path.join(self.model_save_dir, filename)
        
        # Save the model
        self.model.save(filepath)
        print(f"Saved {model_type} model for stage {stage} to {filepath}")
        
        # Store model path in metadata
        if stage not in self.training_metadata["stages"]:
            self.training_metadata["stages"][stage] = {}
        
        self.training_metadata["stages"][stage][f"{model_type}_model_path"] = filepath
        
        # Also save a version without timestamp for easier reference when resuming
        standard_filename = f"{self.model_name}_stage_{stage}_{model_type}.keras"
        standard_filepath = os.path.join(self.model_save_dir, standard_filename)
        self.model.save(standard_filepath)
        
        # Save metadata after each model save
        self._save_metadata()
        
        return filepath
    
    def _save_metadata(self):
        """Save training metadata to JSON file"""
        metadata_path = os.path.join(
            self.model_save_dir, 
            f"training_metadata_{self.timestamp}.json"
        )
        
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=4)
            
        # Also save a latest version for easy reference
        latest_path = os.path.join(self.model_save_dir, "latest_training_metadata.json")
        with open(latest_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=4)
    
    def create_callbacks(self, stage):
        """Create callbacks for the training process"""
        callbacks = []
        
        # Model checkpoint to save the best model
        best_model_path = os.path.join(
            self.model_save_dir, 
            f"{self.model_name}_stage_{stage}_best.keras"  # Changed from .h5 to .keras
        )
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        callbacks.append(reduce_lr)
        
        # Ensure metadata for the stage exists
        if stage not in self.training_metadata["stages"]:
            self.training_metadata["stages"][stage] = {}
            
        # Store checkpoint path in metadata
        self.training_metadata["stages"][stage]["best_model_checkpoint_path"] = best_model_path
        
        return callbacks
    
    def train_stage_1(self, epochs=5):
        """
        Stage 1: Train only the classification head (EfficientNet) or all layers (custom CNN).
        
        Args:
            epochs (int): Number of epochs to train
            
        Returns:
            History: Training history
        """
        if self.model_type == 'efficientnet':
            print("\n=== Stage 1: Training classification head ===")
            # Freeze all base model layers
            self.model_class.freeze_base_model()
            
            # Compile with initial learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            print("\n=== Stage 1: Training custom CNN ===")
            # Compile custom CNN with initial learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Create callbacks
        callbacks = self.create_callbacks(stage=1)
        
        # Ensure metadata for stage 1 exists
        if 1 not in self.training_metadata["stages"]:
            self.training_metadata["stages"][1] = {}
        
        # Store stage-specific metadata
        self.training_metadata["stages"][1] = {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "frozen_layers": "All base model layers" if self.model_type == 'efficientnet' else "None",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Train model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # Update metadata with results
        self.training_metadata["stages"][1]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.training_metadata["stages"][1]["final_val_accuracy"] = float(history.history['val_accuracy'][-1])
        self.training_metadata["stages"][1]["final_val_loss"] = float(history.history['val_loss'][-1])
        
        # Save final model
        self._save_model(stage=1, is_best=False)
        
        # Also load and save the best model as well
        best_model_path = self.training_metadata["stages"][1].get("best_model_checkpoint_path")
        if best_model_path and os.path.exists(best_model_path):
            self._save_model(stage=1, is_best=True)
        
        return history
    
    def train_stage_2(self, epochs=10):
        """
        Stage 2: Fine-tune top 30% of the base model layers (EfficientNet only).
        For custom CNN, continue training all layers.
        
        Args:
            epochs (int): Number of epochs to train
            
        Returns:
            History: Training history
        """
        if self.model_type == 'efficientnet':
            print("\n=== Stage 2: Fine-tuning top 30% of base model ===")
            # Unfreeze top 30% of base model for EfficientNet
            self.model_class.unfreeze_top_layers(percentage=0.3)
            
            # Compile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            print("\n=== Stage 2: Continuing custom CNN training ===")
            # For custom CNN, continue training with a lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=5e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Create callbacks
        callbacks = self.create_callbacks(stage=2)
        
        # Store stage-specific metadata
        self.training_metadata["stages"][2] = {
            "epochs": epochs,
            "learning_rate": 1e-4 if self.model_type == 'efficientnet' else 5e-4,
            "frozen_layers": "70% of base model layers" if self.model_type == 'efficientnet' else "None",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Train model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # Update metadata with results
        self.training_metadata["stages"][2]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.training_metadata["stages"][2]["final_val_accuracy"] = float(history.history['val_accuracy'][-1])
        self.training_metadata["stages"][2]["final_val_loss"] = float(history.history['val_loss'][-1])
        
        # Save final model
        self._save_model(stage=2, is_best=False)
        
        # Also load and save the best model as well
        best_model_path = self.training_metadata["stages"][2].get("best_model_checkpoint_path")
        if best_model_path and os.path.exists(best_model_path):
            self._save_model(stage=2, is_best=True)
        
        return history
    
    def train_stage_3(self, epochs=15):
        """
        Stage 3: Fine-tune all layers of the model with a very low learning rate.
        
        Args:
            epochs (int): Number of epochs to train
            
        Returns:
            History: Training history
        """
        if self.model_type == 'efficientnet':
            print("\n=== Stage 3: Fine-tuning all layers of EfficientNet ===")
            # Unfreeze all layers
            self.model_class.unfreeze_all_layers()
        else:
            print("\n=== Stage 3: Final fine-tuning of custom CNN ===")
        
        # Create callbacks
        callbacks = self.create_callbacks(stage=3)
        
        # Ensure metadata for stage 3 exists
        if 3 not in self.training_metadata["stages"]:
            self.training_metadata["stages"][3] = {}
        
        # Store stage-specific metadata
        self.training_metadata["stages"][3] = {
            "epochs": epochs,
            "learning_rate": 1e-5,
            "frozen_layers": "None",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Train model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # Update metadata with results
        self.training_metadata["stages"][3]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.training_metadata["stages"][3]["final_val_accuracy"] = float(history.history['val_accuracy'][-1])
        self.training_metadata["stages"][3]["final_val_loss"] = float(history.history['val_loss'][-1])
        
        # Save final model
        self._save_model(stage=3, is_best=False)
        
        # Also load and save the best model as well
        best_model_path = self.training_metadata["stages"][3].get("best_model_checkpoint_path")
        if best_model_path and os.path.exists(best_model_path):
            self._save_model(stage=3, is_best=True)
        
        return history
    
    def train_full_pipeline(self, stage1_epochs=5, stage2_epochs=10, stage3_epochs=15, start_stage=1):
        """
        Execute the full progressive training pipeline.
        
        Args:
            stage1_epochs (int): Number of epochs for stage 1
            stage2_epochs (int): Number of epochs for stage 2
            stage3_epochs (int): Number of epochs for stage 3
            start_stage (int): Stage to start training from (1, 2, or 3)
            
        Returns:
            dict: Training history for all stages
        """
        # Prepare data
        self.prepare_data()
        
        # Track overall training metadata
        self.training_metadata["full_pipeline"] = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_stage": start_stage,
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "stage3_epochs": stage3_epochs
        }
        
        # If not starting from stage 1, try to load existing model
        if start_stage > 1:
            previous_stage = start_stage - 1
            model_path = os.path.join(
                self.model_save_dir, 
                f"{self.model_name}_stage_{previous_stage}_final.keras"
            )
            
            # Try to load model from previous stage
            if os.path.exists(model_path):
                print(f"Loading model from previous stage: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                
                # Reconnect model to the model_class for EfficientNet
                if self.model_type == 'efficientnet':
                    self.model_class.model = self.model
            else:
                print(f"Warning: No model found at {model_path}. Creating a new model.")
                self.create_model()
        else:
            # Start from scratch with a new model
            self.create_model()
        
        history = {}
        
        # Execute each training stage as needed
        if start_stage <= 1:
            stage1_history = self.train_stage_1(epochs=stage1_epochs)
            history['stage1'] = stage1_history.history
        
        if start_stage <= 2:
            stage2_history = self.train_stage_2(epochs=stage2_epochs)
            history['stage2'] = stage2_history.history
        
        if start_stage <= 3:
            stage3_history = self.train_stage_3(epochs=stage3_epochs)
            history['stage3'] = stage3_history.history
        
        # Finalize metadata
        self.training_metadata["full_pipeline"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_metadata()
        
        print(f"\n=== {self.model_type.upper()} training completed successfully ===")
        print(f"All models and metadata saved to {self.model_save_dir}")
        
        return history


if __name__ == "__main__":
    # Example usage
    trainer = ProgressiveTrainer(
        train_data_dir='../../data/data_processed/train',
        val_data_dir='../../data/data_processed/val',
        model_save_dir='../../results/models',
        img_size=(512, 512),
        batch_size=16,
        use_class_weights=True
    )
    
    history = trainer.train_full_pipeline(
        stage1_epochs=5,
        stage2_epochs=10,
        stage3_epochs=15
    )