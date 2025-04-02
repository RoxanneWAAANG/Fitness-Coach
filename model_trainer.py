import os
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

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
                meta_file = os.path.join(self.data_dir, f"session_{session_id}_meta.json")
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
    
    def train_tensorflow_model(self, exercise_type):
        """Train a TensorFlow model for form quality prediction"""
        # Load training data
        data = self.load_training_data(exercise_type)
        if data is None or len(data) < 50:  # TF models need more data
            print("Insufficient training data for TensorFlow model")
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
        
        # Convert to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: good, moderate, poor
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_onehot,
            epochs=50,
            batch_size=16,
            validation_data=(X_test_scaled, y_test_onehot),
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test_scaled, y_test_onehot)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model and scaler
        model_file = os.path.join(self.models_dir, f"{exercise_type}_tf_model")
        scaler_file = os.path.join(self.models_dir, f"{exercise_type}_tf_scaler.pkl")
        
        model.save(model_file)
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
            
        # Save label mapping
        label_file = os.path.join(self.models_dir, f"{exercise_type}_tf_labels.json")
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
    
    def fine_tune_pose_model(self, base_model_path, exercise_type):
        """Fine-tune a pose estimation model with your data"""
        # This is more complex and requires TensorFlow model fine-tuning
        # Simplified implementation - would need more code for a full solution
        print("Fine-tuning pose models requires substantial data and compute.")
        print("Consider running this on a more powerful computer than the Raspberry Pi.")
        
        # Placeholder for more complex implementation
        return None