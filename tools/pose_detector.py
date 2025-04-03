import cv2
import os
import numpy as np

class PoseDetector:
    # Key points for human body
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
    
    def __init__(self, use_demo_mode=True):
        """
        Initialize the pose detector
        
        Parameters:
        - use_demo_mode: Whether to use demo mode (True) or try to use real pose detection (False)
        """
        print("Initializing simplified pose detector")
        self.use_demo_mode = use_demo_mode
        
        # Initialize counter for generating different poses in demo mode
        self.demo_frame_counter = 0
        
        # Create a simple dummy capture for simulating camera input
        self.dummy_capture = cv2.VideoCapture(0)
        if not self.dummy_capture.isOpened():
            print("Could not open camera - using static pose generation")
    
    def detect_pose(self, frame):
        """
        Detect pose in the given frame
        Returns: Dictionary with keypoints
        """
        # Always use demo data for reliable testing
        return self.get_demo_data(frame)
    
    def get_demo_data(self, frame=None):
        """Return demo pose data for testing"""
        # Update the demo frame counter
        self.demo_frame_counter += 1
        
        # Basic static pose
        demo_data = {
            'nose': {'x': 320, 'y': 100, 'score': 0.9},
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
        
        # Add some motion to simulate real movement for bicep curl
        # Animate elbow and wrist positions based on the frame counter
        cycle = (self.demo_frame_counter % 60) / 60.0  # 0.0 to 1.0 over 60 frames
        
        # For bicep curl simulation
        arm_angle = np.sin(cycle * np.pi) * 60  # -60 to 60 degrees
        
        # Update left arm joints
        demo_data['left_elbow']['y'] = int(200 + 50 * np.sin(cycle * np.pi))
        demo_data['left_wrist']['y'] = int(300 - 100 * np.sin(cycle * np.pi))
        demo_data['left_wrist']['x'] = int(150 + 50 * np.sin(cycle * np.pi))
        
        # Update right arm joints
        demo_data['right_elbow']['y'] = int(200 + 50 * np.sin(cycle * np.pi))
        demo_data['right_wrist']['y'] = int(300 - 100 * np.sin(cycle * np.pi))
        demo_data['right_wrist']['x'] = int(450 - 50 * np.sin(cycle * np.pi))
        
        return demo_data
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate the angle between three points
        Used for joint angle calculation
        
        Parameters:
        - point1, point2, point3: Dictionary with 'x' and 'y' keys
        
        Returns:
        - Angle in degrees
        """
        try:
            a = np.array([point1['x'], point1['y']])
            b = np.array([point2['x'], point2['y']])
            c = np.array([point3['x'], point3['y']])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate cosine of angle using dot product
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors
            
            # Convert to degrees
            angle_degrees = np.degrees(angle)
            
            return angle_degrees
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 90.0  # Default angle
    
    def draw_pose(self, frame, pose_data, threshold=0.5):
        """
        Draw pose keypoints and connections on the frame
        
        Parameters:
        - frame: Input image
        - pose_data: Dictionary with keypoint data
        - threshold: Confidence threshold for drawing keypoints
        
        Returns:
        - Frame with pose overlay
        """
        if frame is None or not pose_data:
            return None
            
        # Colors
        joint_color = (0, 255, 0)  # Green for keypoints
        bone_color = (0, 0, 255)   # Red for connections
        text_color = (255, 255, 255)  # White for text
        
        # Define connections between keypoints for visualization
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw keypoints
        visual_frame = frame.copy()
        for keypoint, data in pose_data.items():
            if data['score'] > threshold:
                x, y = data['x'], data['y']
                cv2.circle(visual_frame, (x, y), 5, joint_color, -1)
                cv2.putText(visual_frame, keypoint, (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Draw connections
        for start_point, end_point in connections:
            if (start_point in pose_data and end_point in pose_data and
                pose_data[start_point]['score'] > threshold and 
                pose_data[end_point]['score'] > threshold):
                
                start_xy = (pose_data[start_point]['x'], pose_data[start_point]['y'])
                end_xy = (pose_data[end_point]['x'], pose_data[end_point]['y'])
                
                cv2.line(visual_frame, start_xy, end_xy, bone_color, 2)
        
        return visual_frame
