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
    
    def analyze_form(self, exercise_type, pose_data, mpu_data, exercise_params):
        """
        Analyze exercise form based on pose and motion data
        Returns dictionary with form quality, angle corrections, and rep detection
        """
        result = {
            'quality': 'moderate',
            'angle_correction': 0,
            'rep_completed': False
        }
        
        # Check if pose data is valid (key points are detected with sufficient confidence)
        if not self.validate_pose_data(pose_data, exercise_params['keypoints']):
            return result
        
        # Calculate joint angles based on exercise type
        joint_angles = self.calculate_joint_angles(exercise_type, pose_data)
        
        # Check form quality by comparing to target angles
        quality, angle_diff = self.check_form_quality(
            joint_angles, 
            exercise_params['target_angles'],
            exercise_params['tolerance']
        )
        
        # Check if a repetition was performed
        rep_completed = self.detect_repetition(
            exercise_type, 
            joint_angles, 
            mpu_data
        )
        
        # Update result
        result['quality'] = quality
        result['angle_correction'] = angle_diff  # Correction needed in degrees
        result['rep_completed'] = rep_completed
        
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
