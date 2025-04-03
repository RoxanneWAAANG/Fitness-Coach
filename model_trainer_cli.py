#!/usr/bin/env python3
"""
Command-line interface for training models on collected fitness data
"""

import os
import sys
import json
import argparse
from datetime import datetime
from model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train models on collected fitness data")
    parser.add_argument("--exercise", type=str, help="Exercise type to train on (e.g., squat, pushup)")
    parser.add_argument("--model", choices=["random_forest", "pytorch", "unet"], 
                        default="random_forest", help="Model type to train")
    parser.add_argument("--data-dir", type=str, default="collected_data", 
                        help="Directory containing collected data")
    parser.add_argument("--models-dir", type=str, default="models", 
                        help="Directory to save trained models")
    parser.add_argument("--list-exercises", action="store_true", 
                        help="List available exercises in the data")
    parser.add_argument("--session-ids", type=str, 
                        help="Comma-separated list of specific session IDs to use for training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir=args.data_dir, models_dir=args.models_dir)
    
    # List available exercises
    if args.list_exercises:
        list_available_exercises(args.data_dir)
        return
    
    # Check if an exercise is specified
    if not args.exercise:
        print("Error: You must specify an exercise type using --exercise")
        parser.print_help()
        return
    
    # Start training
    print(f"Training {args.model} model for {args.exercise} exercise...")
    
    # Set output streams to capture progress
    original_stdout = sys.stdout
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.exercise}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    try:
        with open(log_file, 'w') as f:
            # Duplicate output to both console and log file
            class TeeOutput:
                def __init__(self, file1, file2):
                    self.file1 = file1
                    self.file2 = file2
                
                def write(self, data):
                    self.file1.write(data)
                    self.file2.write(data)
                    self.file1.flush()
                    self.file2.flush()
                
                def flush(self):
                    self.file1.flush()
                    self.file2.flush()
            
            sys.stdout = TeeOutput(original_stdout, f)
            
            print(f"=== Training started at {datetime.now().isoformat()} ===")
            print(f"Exercise: {args.exercise}")
            print(f"Model type: {args.model}")
            print(f"Data directory: {args.data_dir}")
            print(f"Models directory: {args.models_dir}")
            print(f"Log file: {log_file}")
            print("="*60)
            
            # Train the model
            result = None
            if args.model == "random_forest":
                result = trainer.train_sklearn_model(args.exercise)
            elif args.model == "pytorch":
                result = trainer.train_pytorch_model(args.exercise)
            elif args.model == "unet":
                result = trainer.train_unet_model(args.exercise)
            
            # Show results
            if result:
                print("\n=== Training Results ===")
                print(f"Accuracy: {result.get('accuracy', 'N/A')}")
                print(f"Features used: {len(result.get('feature_names', []))}")
                print("="*60)
                print("Training completed successfully!")
            else:
                print("\n!!! Training Failed !!!")
                print("Check the error messages above for details.")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        print(f"Training complete. Log saved to {log_file}")

def list_available_exercises(data_dir):
    """List all exercises available in the data directory, including CSV files"""
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return
    
    # Get all metadata and CSV files
    meta_files = [f for f in os.listdir(data_dir) 
                  if f.startswith("session_") and f.endswith("_meta.json")]
    csv_files = [f for f in os.listdir(data_dir) 
                 if f.startswith("session_") and f.endswith("_labeled.csv")]
    
    exercises = {}
    
    # Process metadata files
    for meta_file in meta_files:
        try:
            with open(os.path.join(data_dir, meta_file), 'r') as f:
                metadata = json.load(f)
                exercise = metadata.get("exercise", "unknown")
                session_id = meta_file.split("_")[1]
                
                if exercise not in exercises:
                    exercises[exercise] = []
                    
                exercises[exercise].append(session_id)
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")
    
    # Process CSV files
    for csv_file in csv_files:
        try:
            session_id = csv_file.split("_")[1]
            exercise = "unknown"  # Default value if not found
            
            # Attempt to read the exercise type from the CSV file
            with open(os.path.join(data_dir, csv_file), 'r') as f:
                # Assuming the first row contains headers and one of them is 'exercise'
                headers = f.readline().strip().split(',')
                if 'exercise' in headers:
                    exercise_index = headers.index('exercise')
                    first_row = f.readline().strip().split(',')
                    exercise = first_row[exercise_index]

            if exercise not in exercises:
                exercises[exercise] = []

            exercises[exercise].append(session_id)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not exercises:
        print("No exercise data found")
        return
    
    print("\n=== Available Exercises ===")
    print(f"{'Exercise':<15} {'Sessions':<10} {'Session IDs'}")
    print("-"*60)
    
    for exercise, sessions in exercises.items():
        print(f"{exercise:<15} {len(sessions):<10} {', '.join(sessions[:5])}" + 
              (f" and {len(sessions)-5} more..." if len(sessions) > 5 else ""))
    
    print("\nTo train a model, use:")
    print("  python model_trainer_cli.py --exercise <exercise_name> --model <model_type>")
    print("\nExample:")
    print("  python model_trainer_cli.py --exercise squat --model random_forest")

if __name__ == "__main__":
    main() 