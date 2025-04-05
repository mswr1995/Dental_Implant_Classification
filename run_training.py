import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train import ProgressiveTrainer
from src.training.evaluate import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate dental implant classification model')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/data_processed/train',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='data/data_processed/val',
                        help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, default='data/data_processed/test',
                        help='Path to test data directory')
    parser.add_argument('--model_dir', type=str, default='results/models',
                        help='Path to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Path to save evaluation results')
    
    # Model selection and parameters
    parser.add_argument('--model_type', type=str, default='efficientnet',
                        choices=['efficientnet', 'custom_cnn'],
                        help='Type of model to train (efficientnet or custom_cnn)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Target image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    
    # Training parameters
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced classes')
    parser.add_argument('--stage1_epochs', type=int, default=5,
                        help='Number of epochs for stage 1 training')
    parser.add_argument('--stage2_epochs', type=int, default=10,
                        help='Number of epochs for stage 2 training')
    parser.add_argument('--stage3_epochs', type=int, default=15,
                        help='Number of epochs for stage 3 training')
    parser.add_argument('--start_stage', type=int, default=1, 
                        choices=[1, 2, 3],
                        help='Stage to start training from (1, 2, or 3)')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and only run evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model for evaluation')
    
    # Memory management parameters
    parser.add_argument('--memory_growth', action='store_true',
                        help='Enable memory growth for GPU')
    parser.add_argument('--memory_limit', type=int, default=None,
                        help='Limit GPU memory in MB')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (fp16)')
    
    return parser.parse_args()

def configure_gpu(memory_growth=False, memory_limit=None, mixed_precision=False):
    """
    Configure GPU settings to manage memory usage.
    
    Args:
        memory_growth (bool): Whether to enable memory growth
        memory_limit (int): Memory limit in MB, or None for no limit
        mixed_precision (bool): Whether to use mixed precision training
    """
    # Configure memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu}")
                
                if memory_limit:
                    memory_limit_bytes = memory_limit * 1024 * 1024  # Convert MB to bytes
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_bytes)]
                    )
                    print(f"Memory limit set to {memory_limit}MB for {gpu}")
            
            print(f"GPUs available: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs detected. Running on CPU.")
    
    # Configure mixed precision if requested
    if mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled (float16)")
        except Exception as e:
            print(f"Failed to enable mixed precision: {e}")

def main():
    args = parse_args()
    
    # Configure GPU settings
    configure_gpu(
        memory_growth=args.memory_growth,
        memory_limit=args.memory_limit,
        mixed_precision=args.mixed_precision
    )
    
    print("=" * 50)
    print(f"Dental Implant Classification with {args.model_type.upper()}")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {args.model_type}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Use class weights: {args.use_class_weights}")
    if args.mixed_precision:
        print("Using mixed precision (fp16)")
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Determine model name based on model type
    model_name = 'efficientnetb3' if args.model_type == 'efficientnet' else 'custom_cnn'
    
    if not args.eval_only:
        print("\nStarting training pipeline...")
        
        # Create trainer with appropriate model type
        trainer = ProgressiveTrainer(
            train_data_dir=args.train_dir,
            val_data_dir=args.val_dir,
            model_save_dir=args.model_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            use_class_weights=args.use_class_weights,
            model_type=args.model_type
        )
        
        # Run training
        history = trainer.train_full_pipeline(
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            start_stage=args.start_stage
        )
        
        # Use the final model for evaluation
        model_path = os.path.join(args.model_dir, f'{model_name}_stage_3_final.keras')
    else:
        print("\nSkipping training, running evaluation only...")
        # Use provided model path for evaluation
        model_path = args.model_path
        if not model_path:
            print("Error: Model path must be provided with --model_path when using --eval_only")
            sys.exit(1)
    
    print("\nStarting model evaluation...")
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        test_data_dir=args.test_dir,
        results_dir=args.results_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    print("\nTraining and evaluation completed successfully!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()