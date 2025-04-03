import os
import json
import pickle
import sys
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# python model_trainer.py --exercise squat --model random_forest

class FitnessDataset(Dataset):
    """Dataset for fitness form data"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FormQualityModel(nn.Module):
    """PyTorch model for form quality prediction"""
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(FormQualityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class ModelTrainer:
    def __init__(self, data_dir="collected_data", models_dir="models"):
        """Initialize model trainer"""
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def load_training_data(self, exercise_type=None):
        """Load all labeled data for training"""
        labeled_dir = os.path.join(self.data_dir, "labeled_data")
        if not os.path.exists(labeled_dir):
            print("No labeled data directory found")
            return None
            
        # Get all labeled data files
        data_files = [f for f in os.listdir(labeled_dir) if f.endswith("_labeled.csv")]
        
        if not data_files:
            print("No labeled data files found")
            return None
            
        # Filter by exercise type if specified
        if exercise_type:
            # Load metadata to check exercise type
            filtered_files = []
            for file in data_files:
                session_id = file.split("_")[1]
                process_id = file.split("_")[2]
                meta_file = os.path.join(self.data_dir, f"session_{session_id}_{process_id}_meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get("exercise") == exercise_type:
                            filtered_files.append(file)
            data_files = filtered_files
            
        if not data_files:
            print(f"No labeled data files found for exercise: {exercise_type}")
            return None
            
        # Load and combine all data
        all_data = []
        for file in data_files:
            file_path = os.path.join(labeled_dir, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
            
        if not all_data:
            return None
            
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} training examples")
        
        return combined_data
    
    def train_sklearn_model(self, exercise_type, model_type="random_forest"):
        """Train a scikit-learn model for form quality prediction"""
        # Load training data
        data = self.load_training_data(exercise_type)
        if data is None or len(data) < 10:
            print("Insufficient training data")
            return None
            
        # Prepare features and labels
        X = data.drop(["label", "timestamp"], axis=1, errors='ignore')
        y = data["label"]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            # Add other model types as needed
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(report)
        
        # Save model and scaler
        model_file = os.path.join(self.models_dir, f"{exercise_type}_{model_type}_model.pkl")
        scaler_file = os.path.join(self.models_dir, f"{exercise_type}_{model_type}_scaler.pkl")
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"Model saved to {model_file}")
        
        return {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "feature_names": list(X.columns)
        }
    
    def train_pytorch_model(self, exercise_type):
        """Train a PyTorch model for form quality prediction"""
        if not PYTORCH_AVAILABLE:
            print("PyTorch is not available. Cannot train PyTorch model.")
            return None
            
        # Load training data
        data = self.load_training_data(exercise_type)
        if data is None or len(data) < 50:  # Need enough data
            print("Insufficient training data for PyTorch model")
            return None
            
        # Prepare features and labels
        X = data.drop(["label", "timestamp"], axis=1, errors='ignore')
        
        # Convert string labels to integers
        label_mapping = {"good": 0, "moderate": 1, "poor": 2}
        y = data["label"].map(label_mapping)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create datasets and dataloaders
        train_dataset = FitnessDataset(X_train_scaled, y_train.values)
        test_dataset = FitnessDataset(X_test_scaled, y_test.values)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Initialize model
        input_size = X_train_scaled.shape[1]
        model = FormQualityModel(input_size)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Print statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Save model and scaler
        model_file = os.path.join(self.models_dir, f"{exercise_type}_pytorch_model.pt")
        scaler_file = os.path.join(self.models_dir, f"{exercise_type}_pytorch_scaler.pkl")
        label_file = os.path.join(self.models_dir, f"{exercise_type}_pytorch_labels.json")
        
        torch.save(model.state_dict(), model_file)
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
            
        with open(label_file, 'w') as f:
            json.dump(label_mapping, f)
            
        print(f"Model saved to {model_file}")
        
        return {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "feature_names": list(X.columns),
            "label_mapping": label_mapping
        }

def main():
    parser = argparse.ArgumentParser(description="Train models on collected fitness data")
    parser.add_argument("--exercise", type=str, help="Exercise type to train on (e.g., squat, pushup, bicep_curl)")
    parser.add_argument("--model", choices=["random_forest", "pytorch"], 
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