import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="collected_data"):
        """Initialize data collector"""
        self.data_dir = data_dir
        self.current_session = None
        self.current_exercise = None
        self.session_data = []
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            os.makedirs(os.path.join(data_dir, "pose_data"))
            os.makedirs(os.path.join(data_dir, "mpu_data"))
            os.makedirs(os.path.join(data_dir, "labeled_data"))
    
    def start_session(self, exercise_name, user_info=None):
        """Start a new data collection session"""
        self.current_exercise = exercise_name
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = []
        
        # Create session metadata
        metadata = {
            "exercise": exercise_name,
            "start_time": datetime.now().isoformat(),
            "user_info": user_info or {},
            "session_id": self.current_session
        }
        
        # Save metadata
        with open(os.path.join(self.data_dir, f"session_{self.current_session}_meta.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return self.current_session
    
    def record_data_point(self, pose_data, mpu_data, form_quality, rep_count, timestamp=None):
        """Record a single data point during exercise"""
        if not self.current_session:
            return False
            
        timestamp = timestamp or time.time()
        
        # Create data point
        data_point = {
            "timestamp": timestamp,
            "pose_data": pose_data,
            "mpu_data": mpu_data,
            "form_quality": form_quality,
            "rep_count": rep_count
        }
        
        self.session_data.append(data_point)
        
        # Periodically save data to avoid losing everything if system crashes
        if len(self.session_data) % 50 == 0:
            self.save_session_data()
            
        return True
    
    def save_session_data(self):
        """Save all collected data from current session"""
        if not self.current_session or not self.session_data:
            return False
            
        # Save pose data
        pose_file = os.path.join(self.data_dir, "pose_data", f"session_{self.current_session}_pose.json")
        pose_data = [{"timestamp": d["timestamp"], "pose": d["pose_data"], 
                       "form_quality": d["form_quality"], "rep_count": d["rep_count"]} 
                     for d in self.session_data]
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f)
            
        # Save MPU data
        mpu_file = os.path.join(self.data_dir, "mpu_data", f"session_{self.current_session}_mpu.csv")
        mpu_rows = []
        for d in self.session_data:
            if d["mpu_data"] and "accel" in d["mpu_data"] and "gyro" in d["mpu_data"]:
                row = {
                    "timestamp": d["timestamp"],
                    "accel_x": d["mpu_data"]["accel"]["x"],
                    "accel_y": d["mpu_data"]["accel"]["y"],
                    "accel_z": d["mpu_data"]["accel"]["z"],
                    "gyro_x": d["mpu_data"]["gyro"]["x"],
                    "gyro_y": d["mpu_data"]["gyro"]["y"],
                    "gyro_z": d["mpu_data"]["gyro"]["z"],
                    "magnitude": d["mpu_data"].get("magnitude", 0),
                    "form_quality": d["form_quality"],
                    "rep_count": d["rep_count"]
                }
                mpu_rows.append(row)
        
        if mpu_rows:
            pd.DataFrame(mpu_rows).to_csv(mpu_file, index=False)
            
        # Create labeled dataset for training
        labeled_file = os.path.join(self.data_dir, "labeled_data", f"session_{self.current_session}_labeled.csv")
        self.create_labeled_dataset(labeled_file)
        
        return True
    
    def create_labeled_dataset(self, output_file):
        """Create a labeled dataset suitable for model training"""
        if not self.session_data:
            return False
            
        labeled_rows = []
        
        for d in self.session_data:
            # Extract features from pose data
            pose_features = self.extract_pose_features(d["pose_data"])
            
            # Extract features from MPU data
            mpu_features = self.extract_mpu_features(d["mpu_data"])
            
            # Combine features
            features = {**pose_features, **mpu_features}
            
            # Add label
            features["label"] = d["form_quality"]
            features["timestamp"] = d["timestamp"]
            
            labeled_rows.append(features)
            
        # Save to CSV
        if labeled_rows:
            pd.DataFrame(labeled_rows).to_csv(output_file, index=False)
            
        return True
    
    def extract_pose_features(self, pose_data):
        """Extract useful features from pose data for model training"""
        features = {}
        
        if not pose_data:
            # Return empty features if no pose data
            return features
            
        # Extract joint angles (if keypoints exist)
        try:
            # Example: Calculate knee angle if points exist
            if all(k in pose_data for k in ["left_knee", "left_hip", "left_ankle"]):
                # Calculate vectors
                hip = np.array([pose_data["left_hip"]["x"], pose_data["left_hip"]["y"]])
                knee = np.array([pose_data["left_knee"]["x"], pose_data["left_knee"]["y"]])
                ankle = np.array([pose_data["left_ankle"]["x"], pose_data["left_ankle"]["y"]])
                
                # Calculate vectors
                v1 = hip - knee
                v2 = ankle - knee
                
                # Calculate angle
                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
                features["left_knee_angle"] = angle
            
            # Similar calculations for other joints
            # ...
            
            # Add keypoint confidence scores
            for keypoint, data in pose_data.items():
                features[f"{keypoint}_conf"] = data.get("score", 0)
                
        except Exception as e:
            print(f"Error extracting pose features: {e}")
            
        return features
    
    def extract_mpu_features(self, mpu_data):
        """Extract useful features from MPU data for model training"""
        features = {}
        
        if not mpu_data or "accel" not in mpu_data:
            # Return empty features if no MPU data
            return features
            
        try:
            # Add raw accelerometer and gyroscope values
            for axis in ["x", "y", "z"]:
                features[f"accel_{axis}"] = mpu_data["accel"][axis]
                if "gyro" in mpu_data:
                    features[f"gyro_{axis}"] = mpu_data["gyro"][axis]
            
            # Add magnitude
            if "magnitude" in mpu_data:
                features["accel_magnitude"] = mpu_data["magnitude"]
                
        except Exception as e:
            print(f"Error extracting MPU features: {e}")
            
        return features
    
    def end_session(self):
        """End the current data collection session"""
        if not self.current_session:
            return False
            
        # Save final data
        self.save_session_data()
        
        # Update metadata with end time
        metadata_file = os.path.join(self.data_dir, f"session_{self.current_session}_meta.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            metadata["end_time"] = datetime.now().isoformat()
            metadata["data_points"] = len(self.session_data)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Reset session
        self.current_session = None
        self.session_data = []
        
        return True