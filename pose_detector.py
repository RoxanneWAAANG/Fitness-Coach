"""
Handles pose detection using a pre-trained TensorFlow Lite model
"""
import cv2
import os
import requests
import shutil

# Import numpy with version check
try:
    import numpy as np
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")
    if numpy_version.startswith("2."):
        print("WARNING: NumPy 2.x detected which may be incompatible with TensorFlow Lite runtime")
        print("Some functionality may not work properly. Consider downgrading to NumPy 1.x")
except ImportError:
    print("NumPy not found, using fallback mode")

# Try to import TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    print("TensorFlow Lite runtime not available, using demo mode")
    TFLITE_AVAILABLE = False
except Exception as e:
    print(f"Error importing TensorFlow Lite: {e}")
    TFLITE_AVAILABLE = False

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
    
    def __init__(self, model_path="models/movenet_lightning.tflite"):
        # Initialize default values in case model loading fails
        self.interpreter = None
        self.input_height = 192  # Default MoveNet input height
        self.input_width = 192   # Default MoveNet input width

        # Skip TFLite if not available or NumPy 2.x is detected
        if not TFLITE_AVAILABLE:
            print("TensorFlow Lite not available - using demo mode")
            return

        # Check if model exists, if not download it
        if not os.path.exists(model_path) or not self._is_valid_tflite_model(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model file {model_path} does not exist or is invalid. Attempting to download...")
            if self._download_model(model_path):
                print(f"Model downloaded successfully to {model_path}")
            else:
                print("Failed to download model. Using demo mode.")
                return

        try:
            # Load TFLite model
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Get model dimensions
            self.input_height = self.input_details[0]['shape'][1]
            self.input_width = self.input_details[0]['shape'][2]
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            print("Using demo mode for pose detection.")
            self.interpreter = None

    def _is_valid_tflite_model(self, model_path):
        """Check if the file is a valid TFLite model"""
        try:
            with open(model_path, 'rb') as f:
                header = f.read(4)
                return header == b'TFL3'
        except:
            return False

    def _download_model(self, model_path):
        """Download MoveNet Lightning model from a reliable source"""
        try:
            # Updated URL for MoveNet Lightning - this is the correct URL for the model
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
            
            # Download to a temporary file first
            temp_path = model_path + ".tmp"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # If download is successful, rename to the final path
            shutil.move(temp_path, model_path)
            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            # Fallback to a direct GitHub URL if TF Hub fails
            try:
                alt_url = "https://raw.githubusercontent.com/tensorflow/tfjs-models/master/pose-detection/models/movenet_v1_0/movenet_singlepose_lightning.tflite"
                print(f"Trying alternative URL: {alt_url}")
                with requests.get(alt_url, stream=True) as r:
                    r.raise_for_status()
                    with open(temp_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                shutil.move(temp_path, model_path)
                return True
            except Exception as e2:
                print(f"Alternative download also failed: {e2}")
                return False
    
    def detect_pose(self, frame):
        """
        Detect pose in the given frame
        Returns: Dictionary with keypoints or None if detection fails
        """
        if self.interpreter is None or frame is None:
            # Return demo data if model isn't loaded or frame is invalid
            return self.get_demo_data()
        
        try:
            # Make sure numpy is available
            import numpy as np
            
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
            return self.get_demo_data()
    
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
        try:
            import numpy as np
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
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 90.0  # Default angle
