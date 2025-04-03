"""
Analyzes exercise form and counts repetitions
"""
import numpy as np
import time

class ExerciseAnalyzer:
    def __init__(self):
        # Repetition detection parameters
        self.rep_state = "up"  # or "down"
        self.last_angle = None
        self.angle_history = []
        self.rep_threshold = 20  # degrees
        self.last_rep_time = 0
        self.min_rep_interval = 1.0  # seconds
    
    # def analyze_form(self, exercise_type, pose_data, mpu_data, exercise_params):
    #     """
    #     Analyze exercise form based on pose and motion data
    #     Returns dictionary with form quality, angle corrections, and rep detection
    #     """
    #     result = {
    #         'quality': 'moderate',
    #         'angle_correction': 0,
    #         'rep_completed': False
    #     }
        
    #     # Check if pose data is valid (key points are detected with sufficient confidence)
    #     if not self.validate_pose_data(pose_data, exercise_params['keypoints']):
    #         return result
        
    #     # Calculate joint angles based on exercise type
    #     joint_angles = self.calculate_joint_angles(exercise_type, pose_data)
        
    #     # Check form quality by comparing to target angles
    #     quality, angle_diff = self.check_form_quality(
    #         joint_angles, 
    #         exercise_params['target_angles'],
    #         exercise_params['tolerance']
    #     )
        
    #     # Check if a repetition was performed
    #     rep_completed = self.detect_repetition(
    #         exercise_type, 
    #         joint_angles, 
    #         mpu_data
    #     )
        
    #     # Update result
    #     result['quality'] = quality
    #     result['angle_correction'] = angle_diff  # Correction needed in degrees
    #     result['rep_completed'] = rep_completed
        
    #     return result
    def analyze_form(self, exercise_type, pose_data, mpu_data, exercise_params):
        """
        Analyze exercise form based on pose and motion data
        First tries to use a custom model if available
        """
        # First try to load and use U-Net model
        unet_model = self.load_unet_model(exercise_type)
        if unet_model is not None:
            return self.analyze_form_with_unet(exercise_type, pose_data, mpu_data, unet_model)
        
        # Then try PyTorch model
        pytorch_model = self.load_pytorch_model(exercise_type)
        if pytorch_model is not None:
            return self.analyze_form_with_pytorch_model(exercise_type, pose_data, mpu_data, pytorch_model)
        
        # Then try sklearn model
        sklearn_model = self.load_custom_model(exercise_type)
        if sklearn_model is not None:
            return self.analyze_form_with_custom_model(exercise_type, pose_data, mpu_data, sklearn_model)
        
        # Fall back to rule-based analysis
        result = {
            'quality': 'moderate',
            'angle_correction': 0,
            'rep_completed': False
        }
        
        return result
    
    def validate_pose_data(self, pose_data, required_keypoints, confidence_threshold=0.5):
        """Check if required keypoints are detected with sufficient confidence"""
        if not pose_data:
            return False
            
        for keypoint in required_keypoints:
            # For simplicity, we check either left or right keypoints
            left_key = f"left_{keypoint}"
            right_key = f"right_{keypoint}"
            
            left_valid = (left_key in pose_data and pose_data[left_key]['score'] >= confidence_threshold)
            right_valid = (right_key in pose_data and pose_data[right_key]['score'] >= confidence_threshold)
            
            # For center keypoints like nose
            center_valid = (keypoint in pose_data and pose_data[keypoint]['score'] >= confidence_threshold)
            
            if not (left_valid or right_valid or center_valid):
                return False
                
        return True
    
    def calculate_joint_angles(self, exercise_type, pose_data):
        """Calculate relevant joint angles for the specific exercise"""
        joint_angles = {}
        
        if exercise_type == 'squat':
            # Calculate knee angle (ankle-knee-hip)
            if all(k in pose_data for k in ['left_ankle', 'left_knee', 'left_hip']):
                joint_angles['left_knee'] = self.calculate_angle_three_points(
                    pose_data['left_ankle'],
                    pose_data['left_knee'],
                    pose_data['left_hip']
                )
                
            if all(k in pose_data for k in ['right_ankle', 'right_knee', 'right_hip']):
                joint_angles['right_knee'] = self.calculate_angle_three_points(
                    pose_data['right_ankle'],
                    pose_data['right_knee'],
                    pose_data['right_hip']
                )
                
            # Calculate hip angle (knee-hip-shoulder)
            if all(k in pose_data for k in ['left_knee', 'left_hip', 'left_shoulder']):
                joint_angles['left_hip'] = self.calculate_angle_three_points(
                    pose_data['left_knee'],
                    pose_data['left_hip'],
                    pose_data['left_shoulder']
                )
                
            if all(k in pose_data for k in ['right_knee', 'right_hip', 'right_shoulder']):
                joint_angles['right_hip'] = self.calculate_angle_three_points(
                    pose_data['right_knee'],
                    pose_data['right_hip'],
                    pose_data['right_shoulder']
                )
                
        elif exercise_type == 'pushup':
            # Calculate elbow angle (wrist-elbow-shoulder)
            if all(k in pose_data for k in ['left_wrist', 'left_elbow', 'left_shoulder']):
                joint_angles['left_elbow'] = self.calculate_angle_three_points(
                    pose_data['left_wrist'],
                    pose_data['left_elbow'],
                    pose_data['left_shoulder']
                )
                
            if all(k in pose_data for k in ['right_wrist', 'right_elbow', 'right_shoulder']):
                joint_angles['right_elbow'] = self.calculate_angle_three_points(
                    pose_data['right_wrist'],
                    pose_data['right_elbow'],
                    pose_data['right_shoulder']
                )
                
        elif exercise_type == 'bicep_curl':
            # Calculate elbow angle (wrist-elbow-shoulder)
            if all(k in pose_data for k in ['left_wrist', 'left_elbow', 'left_shoulder']):
                joint_angles['left_elbow'] = self.calculate_angle_three_points(
                    pose_data['left_wrist'],
                    pose_data['left_elbow'],
                    pose_data['left_shoulder']
                )
                
            if all(k in pose_data for k in ['right_wrist', 'right_elbow', 'right_shoulder']):
                joint_angles['right_elbow'] = self.calculate_angle_three_points(
                    pose_data['right_wrist'],
                    pose_data['right_elbow'],
                    pose_data['right_shoulder']
                )
        
        return joint_angles
    
    def calculate_angle_three_points(self, point1, point2, point3):
        """Calculate the angle between three points"""
        # Extract coordinates
        x1, y1 = point1['x'], point1['y']
        x2, y2 = point2['x'], point2['y']
        x3, y3 = point3['x'], point3['y']
        
        # Calculate vectors
        vector1 = [x1 - x2, y1 - y2]
        vector2 = [x3 - x2, y3 - y2]
        
        # Calculate dot product
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        
        # Calculate magnitudes
        mag1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        mag2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        
        # Calculate angle in degrees
        angle = np.arccos(dot_product / (mag1 * mag2)) * 180.0 / np.pi
        
        return angle
    
    def check_form_quality(self, joint_angles, target_angles, tolerance):
        """
        Check form quality by comparing measured angles to target angles
        Returns: (quality string, largest angle difference)
        """
        if not joint_angles:
            return 'unknown', 0
            
        max_diff = 0
        
        for joint, target in target_angles.items():
            # Check both left and right sides if available
            left_joint = f"left_{joint}"
            right_joint = f"right_{joint}"
            
            if left_joint in joint_angles:
                diff = abs(joint_angles[left_joint] - target)
                max_diff = max(max_diff, diff)
                
            if right_joint in joint_angles:
                diff = abs(joint_angles[right_joint] - target)
                max_diff = max(max_diff, diff)
        
        # Determine quality based on maximum difference
        if max_diff <= tolerance / 2:
            return 'good', max_diff
        elif max_diff <= tolerance:
            return 'moderate', max_diff
        else:
            return 'poor', max_diff
    
    def detect_repetition(self, exercise_type, joint_angles, mpu_data):
        """
        Detect if a repetition was performed
        Uses both joint angles and accelerometer data
        """
        # Use MPU6050 data for more accurate rep counting if available
        if mpu_data and 'magnitude' in mpu_data:
            # Check for significant motion
            if len(self.angle_history) > 5:  # Need some history to detect pattern
                current_mag = mpu_data['magnitude']
                
                # Detect peak in acceleration
                if current_mag > 1.5 and self.rep_state == "up":
                    # Check time since last rep to avoid double counting
                    current_time = time.time()
                    if current_time - self.last_rep_time > self.min_rep_interval:
                        self.rep_state = "down"
                        self.last_rep_time = current_time
                        return True
                
                # Reset state when movement slows down
                if current_mag < 1.2 and self.rep_state == "down":
                    self.rep_state = "up"
        
        # Fallback to angle-based detection if MPU data isn't reliable
        primary_angle = None
        
        # Select the primary angle to track based on exercise
        if exercise_type == 'squat' and 'right_knee' in joint_angles:
            primary_angle = joint_angles['right_knee']
        elif exercise_type == 'pushup' and 'right_elbow' in joint_angles:
            primary_angle = joint_angles['right_elbow']
        elif exercise_type == 'bicep_curl' and 'right_elbow' in joint_angles:
            primary_angle = joint_angles['right_elbow']
        
        # Add angle to history
        if primary_angle is not None:
            self.angle_history.append(primary_angle)
            if len(self.angle_history) > 20:
                self.angle_history.pop(0)
                
            # Detect repetition pattern
            if len(self.angle_history) >= 10:
                min_angle = min(self.angle_history[-10:])
                max_angle = max(self.angle_history[-10:])
                angle_range = max_angle - min_angle
                
                current_angle = self.angle_history[-1]
                prev_angle = self.angle_history[-2]
                
                # Detect transition from minimum to rising angle
                if angle_range > self.rep_threshold and \
                   prev_angle <= min_angle + 5 and \
                   current_angle > prev_angle + 5 and \
                   self.rep_state == "down":
                    
                    current_time = time.time()
                    if current_time - self.last_rep_time > self.min_rep_interval:
                        self.rep_state = "up"
                        self.last_rep_time = current_time
                        return True
                
                # Detect transition to lowest position
                if angle_range > self.rep_threshold and \
                   prev_angle > current_angle and \
                   current_angle <= min_angle + 5 and \
                   self.rep_state == "up":
                    self.rep_state = "down"
        
        return False

    def load_custom_model(self, exercise_type):
        """Load a custom-trained model for form analysis"""
        try:
            model_file = f"models/{exercise_type}_random_forest_model.pkl"
            scaler_file = f"models/{exercise_type}_random_forest_scaler.pkl"
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                print(f"Custom model for {exercise_type} not found")
                return None
                
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
                
            return {"model": model, "scaler": scaler}
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return None

    def analyze_form_with_custom_model(self, exercise_type, pose_data, mpu_data, model_data):
        """Analyze form using a custom-trained model"""
        result = {
            'quality': 'moderate',
            'angle_correction': 0,
            'rep_completed': False
        }
        
        if not model_data or "model" not in model_data or "scaler" not in model_data:
            # Fall back to standard analysis if model isn't available
            return self.analyze_form(exercise_type, pose_data, mpu_data, exercise_params)
            
        try:
            # Extract features from data
            pose_features = self.extract_features_from_pose(pose_data)
            mpu_features = self.extract_features_from_mpu(mpu_data)
            
            # Combine features
            features = {**pose_features, **mpu_features}
            
            # Convert to dataframe
            df = pd.DataFrame([features])
            
            # Fill missing values
            df = df.fillna(0)
            
            # Scale features
            X_scaled = model_data["scaler"].transform(df)
            
            # Make prediction
            quality = model_data["model"].predict(X_scaled)[0]
            
            # Get correction angle from feature analysis
            correction = self.estimate_correction_angle(exercise_type, pose_data)
            
            # Check for rep completion
            rep_completed = self.detect_repetition(exercise_type, {}, mpu_data)
            
            # Update result
            result['quality'] = quality
            result['angle_correction'] = correction
            result['rep_completed'] = rep_completed
            
            return result
            
        except Exception as e:
            print(f"Error using custom model: {e}")
            # Fall back to standard analysis if prediction fails
            return self.analyze_form(exercise_type, pose_data, mpu_data, exercise_params)

    def load_unet_model(self, exercise_type):
        """Load a trained U-Net model for form analysis"""
        try:
            import torch
            import torch.nn as nn
            
            # Must define the model architecture to match training
            class SimpleUNet(nn.Module):
                def __init__(self, input_features, num_classes=3):
                    super(SimpleUNet, self).__init__()
                    
                    # Number of features at each level
                    base_filters = 16
                    
                    # Encoder path
                    self.enc1 = nn.Sequential(
                        nn.Linear(input_features, base_filters),
                        nn.BatchNorm1d(base_filters),
                        nn.ReLU()
                    )
                    
                    self.enc2 = nn.Sequential(
                        nn.Linear(base_filters, base_filters * 2),
                        nn.BatchNorm1d(base_filters * 2),
                        nn.ReLU()
                    )
                    
                    # Bottleneck
                    self.bottleneck = nn.Sequential(
                        nn.Linear(base_filters * 2, base_filters * 4),
                        nn.BatchNorm1d(base_filters * 4),
                        nn.ReLU(),
                        nn.Linear(base_filters * 4, base_filters * 2),
                        nn.BatchNorm1d(base_filters * 2),
                        nn.ReLU()
                    )
                    
                    # Decoder path
                    self.dec2 = nn.Sequential(
                        nn.Linear(base_filters * 4, base_filters * 2),  # Concatenated input
                        nn.BatchNorm1d(base_filters * 2),
                        nn.ReLU()
                    )
                    
                    self.dec1 = nn.Sequential(
                        nn.Linear(base_filters * 3, base_filters),  # Concatenated input
                        nn.BatchNorm1d(base_filters),
                        nn.ReLU()
                    )
                    
                    # Final layer
                    self.final = nn.Linear(base_filters, num_classes)
                    
                def forward(self, x):
                    # Encoder
                    enc1_out = self.enc1(x)
                    enc2_out = self.enc2(enc1_out)
                    
                    # Bottleneck
                    bottleneck_out = self.bottleneck(enc2_out)
                    
                    # Decoder with skip connections
                    dec2_input = torch.cat([bottleneck_out, enc2_out], dim=1)
                    dec2_out = self.dec2(dec2_input)
                    
                    dec1_input = torch.cat([dec2_out, enc1_out], dim=1)
                    dec1_out = self.dec1(dec1_input)
                    
                    # Final output
                    output = self.final(dec1_out)
                    
                    return output
            
            model_file = f"models/{exercise_type}_unet_model.pt"
            scaler_file = f"models/{exercise_type}_unet_scaler.pkl"
            label_file = f"models/{exercise_type}_unet_labels.json"
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                print(f"U-Net model for {exercise_type} not found")
                return None
                
            # Load scaler
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load label mapping
            with open(label_file, 'r') as f:
                label_mapping = json.load(f)
                
            # Get feature count from scaler
            n_features = scaler.n_features_in_
                
            # Initialize model
            model = SimpleUNet(input_features=n_features)
            
            # Load model weights
            model.load_state_dict(torch.load(model_file))
            model.eval()  # Set to evaluation mode
                
            return {
                "model": model,
                "scaler": scaler,
                "label_mapping": label_mapping,
                "type": "unet"
            }
            
        except Exception as e:
            print(f"Error loading U-Net model: {e}")
            return None

    def analyze_form_with_unet(self, exercise_type, pose_data, mpu_data, model_data):
        """Analyze form using a trained U-Net model"""
        result = {
            'quality': 'moderate',
            'angle_correction': 0,
            'rep_completed': False
        }
        
        if not model_data or "model" not in model_data or "scaler" not in model_data:
            # Fall back to standard analysis if model isn't available
            return self.analyze_form(exercise_type, pose_data, mpu_data, exercise_params)
            
        try:
            import torch
            import numpy as np
            
            # Extract features from data
            pose_features = self.extract_features_from_pose(pose_data)
            mpu_features = self.extract_features_from_mpu(mpu_data)
            
            # Combine features
            features = {**pose_features, **mpu_features}
            
            # Convert to dataframe
            df = pd.DataFrame([features])
            
            # Fill missing values
            df = df.fillna(0)
            
            # Scale features
            X_scaled = model_data["scaler"].transform(df)
            
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            # Make prediction
            model = model_data["model"]
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                
            # Convert class index to label
            label_mapping_inv = {v: k for k, v in model_data["label_mapping"].items()}
            quality = label_mapping_inv[predicted.item()]
            
            # Get correction angle from feature analysis
            correction = self.estimate_correction_angle(exercise_type, pose_data)
            
            # Check for rep completion
            rep_completed = self.detect_repetition(exercise_type, {}, mpu_data)
            
            # Update result
            result['quality'] = quality
            result['angle_correction'] = correction
            result['rep_completed'] = rep_completed
            
            return result
            
        except Exception as e:
            print(f"Error using U-Net model: {e}")
            # Fall back to standard analysis if prediction fails
            return self.analyze_form(exercise_type, pose_data, mpu_data, exercise_params)
