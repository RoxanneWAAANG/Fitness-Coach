"""
Handles pose detection using a pre-trained TensorFlow Lite model
"""
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

class PoseDetector:
    # Key points indices for MoveNet model
    KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    def __init__(self, model_path="models/movenet_lightning_int8.tflite"):
        # Check if model exists, if not download it (you may need to implement this)
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model file {model_path} does not exist. Please download it.")
            print("You can download MoveNet from TensorFlow Hub.")
            # Alternatively, provide a minimal model for testing
            self.interpreter = None
            return
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get model dimensions
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
    
    def detect_pose(self, frame):
        """
        Detect pose in the given frame
        Returns: Dictionary with keypoints or None if detection fails
        """
        if self.interpreter is None:
            # Return demo data if model isn't loaded
            return self.get_demo_data()
        
        try:
            # Preprocess image
            input_image = cv2.resize(frame, (self.input_width, self.input_height))
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = input_image.astype(np.float32)
            input_image = input_image / 255.0
            input_image = np.expand_dims(input_image, axis=0)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
            keypoints = keypoints_with_scores[0, 0, :, :3]
            
            # Format output
            frame_height, frame_width = frame.shape[:2]
            pose_data = {}
            
            for name, idx in self.KEYPOINT_DICT.items():
                y, x, score = keypoints[idx]
                pose_data[name] = {
                    'x': int(x * frame_width),
                    'y': int(y * frame_height),
                    'score': float(score)
                }
            
            return pose_data
            
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None
    
    def get_demo_data(self):
        """Return demo pose data for testing when model isn't available"""
        # Example data for a person standing straight
        demo_data = {
            'nose': {'x': 300, 'y': 100, 'score': 0.9},
            'left_eye': {'x': 290, 'y': 90, 'score': 0.9},
            'right_eye': {'x': 310, 'y': 90, 'score': 0.9},
            'left_ear': {'x': 280, 'y': 100, 'score': 0.8},
            'right_ear': {'x': 320, 'y': 100, 'score': 0.8},
            'left_shoulder': {'x': 250, 'y': 200, 'score': 0.9},
            'right_shoulder': {'x': 350, 'y': 200, 'score': 0.9},
            'left_elbow': {'x': 200, 'y': 250, 'score': 0.9},
            'right_elbow': {'x': 400, 'y': 250, 'score': 0.9},
            'left_wrist': {'x': 150, 'y': 300, 'score': 0.8},
            'right_wrist': {'x': 450, 'y': 300, 'score': 0.8},
            'left_hip': {'x': 275, 'y': 350, 'score': 0.9},
            'right_hip': {'x': 325, 'y': 350, 'score': 0.9},
            'left_knee': {'x': 275, 'y': 450, 'score': 0.9},
            'right_knee': {'x': 325, 'y': 450, 'score': 0.9},
            'left_ankle': {'x': 275, 'y': 550, 'score': 0.8},
            'right_ankle': {'x': 325, 'y': 550, 'score': 0.8}
        }
        print("Using demo pose data for testing")
        return demo_data
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate the angle between three points
        Used for joint angle calculation
        """
        a = np.array([point1['x'], point1['y']])
        b = np.array([point2['x'], point2['y']])
        c = np.array([point3['x'], point3['y']])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        # Convert to degrees
        angle_degrees = np.degrees(angle)
        
        return angle_degrees
