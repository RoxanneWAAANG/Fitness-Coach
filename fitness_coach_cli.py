#!/usr/bin/env python3
"""
fitness_coach_cli.py - Real-time fitness form analyzer and feedback system

This application uses the MoveNet model to analyze workout form in real-time,
provides feedback via LCD display, and allows user interaction through buttons.
It processes data from both camera (pose detection) and MPU6050 sensor.
"""

import os
import time
import json
import numpy as np
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import threading
import cv2
import tflite_runtime.interpreter as tflite
from mpu6050 import mpu6050  # Using the imported MPU6050 module

# Path configuration
MODEL_PATH = "/home/ruoxinwang/aipi590/Fitness_Coach/Fitness-Coach/models"
DATA_DIR = "collected_data"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "pose_data"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "mpu_data"), exist_ok=True)

# GPIO pin configuration
BUTTON_1_PIN = 17  # Navigation button
BUTTON_2_PIN = 27  # Selection button
LED_RED_PIN = 16   # Bad form indicator
LED_GREEN_PIN = 20 # Good form indicator
LED_BLUE_PIN = 21  # Active/processing indicator

# LCD configuration
LCD_I2C_ADDR = 0x27  # Default I2C address for PCF8574 backpack
LCD_COLS = 16
LCD_ROWS = 2

# MPU6050 configuration
MPU_ADDR = 0x53  # Default I2C address for MPU6050

# Exercise types and their keypoints of interest
EXERCISES = {
    'squat': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
    'pushup': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
    'bicep_curl': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
}

# Application state
current_exercise = None
is_recording = False
rep_count = 0
session_data = []
form_quality = "N/A"
menu_position = 0

class PoseDetector:
    """Handles pose detection using TensorFlow Lite MoveNet model"""
    
    # MoveNet keypoint names
    KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_path):
        # Find and load a compatible MoveNet model
        model_files = [f for f in os.listdir(model_path) if f.endswith('.tflite')]
        if not model_files:
            raise FileNotFoundError(f"No TFLite models found in {model_path}")
        
        model_file = os.path.join(model_path, model_files[0])
        print(f"Loading model: {model_file}")
        
        # Initialize TFLite interpreter
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input size
        self.input_size = self.input_details[0]['shape'][1:3]
    
    def detect_pose(self, image):
        """Detect pose in the given image"""
        # Resize and preprocess image
        input_image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
        input_image = np.expand_dims(input_image, axis=0)
        
        # Normalize the image
        input_image = (input_image - 128) / 128
        
        # Set tensor and invoke
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.astype('float32'))
        self.interpreter.invoke()
        
        # Get keypoints
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Process keypoints into a dictionary format
        pose_data = {}
        image_height, image_width = image.shape[:2]
        
        for idx, kp in enumerate(keypoints):
            y, x, score = kp
            # Convert normalized coordinates to actual image coordinates
            x_px = int(x * image_width)
            y_px = int(y * image_height)
            
            # Add to pose data
            pose_data[self.KEYPOINTS[idx]] = {
                'x': x_px,
                'y': y_px,
                'score': float(score)
            }
        
        return pose_data

class ExerciseAnalyzer:
    """Analyzes exercise form and counts repetitions"""
    
    def __init__(self):
        # Thresholds for form detection
        self.form_thresholds = {
            'squat': {
                'knee_angle': (50, 100),  # Min and max angles for good form
                'hip_depth': 0.7  # Relative depth for a good squat
            },
            'pushup': {
                'elbow_angle': (70, 110),  # Min and max angles for good form
                'body_alignment': 0.9  # Threshold for good alignment
            },
            'bicep_curl': {
                'elbow_angle': (40, 160),  # Min and max angles for good form
                'shoulder_stability': 0.8  # Threshold for good stability
            }
        }
        
        self.rep_state = "up"  # Track rep state (up/down)
        self.last_magnitude = 0  # Previous acceleration magnitude
        self.rep_threshold = 0.3  # Threshold for detecting a rep
        self.magnitude_buffer = []  # Buffer for smoothing
    
    def calculate_angle(self, joint1, joint2, joint3):
        """Calculate angle between three joints"""
        # Convert to vectors
        a = np.array([joint1['x'], joint1['y']])
        b = np.array([joint2['x'], joint2['y']])
        c = np.array([joint3['x'], joint3['y']])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle (in degrees)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def analyze_form(self, exercise_type, pose_data, mpu_data):
        """Analyze form quality for the given exercise"""
        # Check if all required keypoints are detected with good confidence
        for keypoint in EXERCISES[exercise_type]:
            if keypoint not in pose_data or pose_data[keypoint]['score'] < 0.5:
                return "poor"  # Cannot determine form if keypoints are missing
        
        # Exercise-specific analysis
        if exercise_type == 'squat':
            # Calculate knee angles
            left_knee_angle = self.calculate_angle(
                pose_data['left_hip'], 
                pose_data['left_knee'], 
                pose_data['left_ankle']
            )
            right_knee_angle = self.calculate_angle(
                pose_data['right_hip'], 
                pose_data['right_knee'], 
                pose_data['right_ankle']
            )
            
            # Calculate hip depth relative to knees
            hip_y = (pose_data['left_hip']['y'] + pose_data['right_hip']['y']) / 2
            knee_y = (pose_data['left_knee']['y'] + pose_data['right_knee']['y']) / 2
            hip_depth = (hip_y - knee_y) / knee_y
            
            # Check if angles and depth are within good form thresholds
            thresholds = self.form_thresholds['squat']
            
            if (thresholds['knee_angle'][0] <= left_knee_angle <= thresholds['knee_angle'][1] and
                thresholds['knee_angle'][0] <= right_knee_angle <= thresholds['knee_angle'][1] and
                hip_depth >= thresholds['hip_depth']):
                return "good"
            elif (thresholds['knee_angle'][0] - 10 <= left_knee_angle <= thresholds['knee_angle'][1] + 10 and
                  thresholds['knee_angle'][0] - 10 <= right_knee_angle <= thresholds['knee_angle'][1] + 10 and
                  hip_depth >= thresholds['hip_depth'] * 0.8):
                return "moderate"
            else:
                return "poor"
                
        elif exercise_type == 'pushup':
            # Calculate elbow angles
            left_elbow_angle = self.calculate_angle(
                pose_data['left_shoulder'], 
                pose_data['left_elbow'], 
                pose_data['left_wrist']
            )
            right_elbow_angle = self.calculate_angle(
                pose_data['right_shoulder'], 
                pose_data['right_elbow'], 
                pose_data['right_wrist']
            )
            
            # Check for body alignment (shoulders and hips should be aligned)
            shoulder_y = (pose_data['left_shoulder']['y'] + pose_data['right_shoulder']['y']) / 2
            hip_y = (pose_data['left_hip']['y'] + pose_data['right_hip']['y']) / 2
            alignment = abs(shoulder_y - hip_y) / shoulder_y
            
            # Check form against thresholds
            thresholds = self.form_thresholds['pushup']
            
            if (thresholds['elbow_angle'][0] <= left_elbow_angle <= thresholds['elbow_angle'][1] and
                thresholds['elbow_angle'][0] <= right_elbow_angle <= thresholds['elbow_angle'][1] and
                alignment <= thresholds['body_alignment']):
                return "good"
            elif (thresholds['elbow_angle'][0] - 10 <= left_elbow_angle <= thresholds['elbow_angle'][1] + 10 and
                  thresholds['elbow_angle'][0] - 10 <= right_elbow_angle <= thresholds['elbow_angle'][1] + 10 and
                  alignment <= thresholds['body_alignment'] * 1.2):
                return "moderate"
            else:
                return "poor"
                
        elif exercise_type == 'bicep_curl':
            # Calculate elbow angles
            left_elbow_angle = self.calculate_angle(
                pose_data['left_shoulder'], 
                pose_data['left_elbow'], 
                pose_data['left_wrist']
            )
            right_elbow_angle = self.calculate_angle(
                pose_data['right_shoulder'], 
                pose_data['right_elbow'], 
                pose_data['right_wrist']
            )
            
            # Check shoulder stability (shoulders shouldn't move much during curl)
            shoulder_stability = 1.0  # Default good value
            if 'accel' in mpu_data:
                # Use MPU data to check for excessive shoulder movement
                shoulder_stability = 1.0 / (1.0 + abs(mpu_data['accel']['y']) * 5)
            
            # Check form against thresholds
            thresholds = self.form_thresholds['bicep_curl']
            
            if (thresholds['elbow_angle'][0] <= left_elbow_angle <= thresholds['elbow_angle'][1] and
                thresholds['elbow_angle'][0] <= right_elbow_angle <= thresholds['elbow_angle'][1] and
                shoulder_stability >= thresholds['shoulder_stability']):
                return "good"
            elif (thresholds['elbow_angle'][0] - 10 <= left_elbow_angle <= thresholds['elbow_angle'][1] + 10 and
                  thresholds['elbow_angle'][0] - 10 <= right_elbow_angle <= thresholds['elbow_angle'][1] + 10 and
                  shoulder_stability >= thresholds['shoulder_stability'] * 0.8):
                return "moderate"
            else:
                return "poor"
        
        return "moderate"  # Default return
    
    def count_repetition(self, mpu_data):
        """Count exercise repetitions using MPU6050 data"""
        if 'magnitude' not in mpu_data:
            return 0  # Cannot count if magnitude is missing
        
        # Add current magnitude to buffer
        self.magnitude_buffer.append(mpu_data['magnitude'])
        if len(self.magnitude_buffer) > 10:
            self.magnitude_buffer.pop(0)
        
        # Smooth magnitude
        magnitude = sum(self.magnitude_buffer) / len(self.magnitude_buffer)
        
        # Detect repetitions using magnitude changes
        rep_increment = 0
        
        if self.rep_state == "up" and magnitude > self.last_magnitude + self.rep_threshold:
            # Transition to "down" state
            self.rep_state = "down"
        elif self.rep_state == "down" and magnitude < self.last_magnitude - self.rep_threshold:
            # Transition to "up" state and increment rep counter
            self.rep_state = "up"
            rep_increment = 1
        
        self.last_magnitude = magnitude
        
        return rep_increment

# Hardware controller class
class HardwareController:
    """Controls hardware components: LCD, LEDs, and buttons"""
    
    def __init__(self):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup buttons as inputs with pull-down resistors
        GPIO.setup(BUTTON_1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(BUTTON_2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Setup LEDs as outputs
        GPIO.setup(LED_RED_PIN, GPIO.OUT)
        GPIO.setup(LED_GREEN_PIN, GPIO.OUT)
        GPIO.setup(LED_BLUE_PIN, GPIO.OUT)
        
        # Initialize LCD
        try:
            self.lcd = CharLCD(i2c_expander='PCF8574', address=LCD_I2C_ADDR, cols=LCD_COLS, rows=LCD_ROWS)
            self.lcd.clear()
            self.lcd.write_string("Fitness Coach\nInitializing...")
        except Exception as e:
            print(f"LCD init error: {e}")
            self.lcd = None
    
    def set_led(self, r, g, b):
        """Set RGB LED state (each value is 0 or 1)"""
        GPIO.output(LED_RED_PIN, r)
        GPIO.output(LED_GREEN_PIN, g)
        GPIO.output(LED_BLUE_PIN, b)
    
    def set_led_by_form(self, form_quality):
        """Set LED color based on form quality"""
        if form_quality == "good":
            self.set_led(0, 1, 0)  # Green
        elif form_quality == "moderate":
            self.set_led(1, 1, 0)  # Yellow (Red + Green)
        elif form_quality == "poor":
            self.set_led(1, 0, 0)  # Red
        else:
            self.set_led(0, 0, 1)  # Blue (inactive/waiting)
    
    def update_lcd(self, line1, line2=""):
        """Update LCD display with two lines of text"""
        if self.lcd:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(line1[:LCD_COLS])
            
            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:LCD_COLS])
    
    def cleanup(self):
        """Clean up GPIO and hardware resources"""
        if self.lcd:
            self.lcd.clear()
            self.lcd.close()
        GPIO.cleanup()

# Main application class
class FitnessCoach:
    """Main application class for the Fitness Coach system"""
    
    def __init__(self):
        print("Initializing Fitness Coach...")
        
        # Initialize hardware
        self.hw = HardwareController()
        
        # Initialize MPU6050 using the imported module
        try:
            self.mpu = mpu6050(MPU_ADDR)
            print("MPU6050 initialized")
        except Exception as e:
            print(f"MPU6050 init error: {e}")
            self.mpu = None
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, _ = self.camera.read()
            if not ret:
                raise Exception("Failed to read from camera")
            print("Camera initialized")
        except Exception as e:
            print(f"Camera init error: {e}")
            self.camera = None
        
        # Initialize pose detector
        try:
            self.pose_detector = PoseDetector(MODEL_PATH)
            print("Pose detector initialized")
        except Exception as e:
            print(f"Pose detector init error: {e}")
            self.pose_detector = None
        
        # Initialize exercise analyzer
        self.analyzer = ExerciseAnalyzer()
        
        # Setup button event listeners
        GPIO.add_event_detect(BUTTON_1_PIN, GPIO.RISING, callback=self.on_button1_press, bouncetime=300)
        GPIO.add_event_detect(BUTTON_2_PIN, GPIO.RISING, callback=self.on_button2_press, bouncetime=300)
        
        # Application state
        self.current_exercise = None
        self.is_recording = False
        self.rep_count = 0
        self.session_data = []
        self.form_quality = "N/A"
        self.menu_position = 0
        self.menu_items = list(EXERCISES.keys()) + ["Exit"]
        
        # Session ID for data saving
        self.session_id = int(time.time())
        
        # Initialize threads
        self.recording_thread = None
        self.is_running = True
    
    def on_button1_press(self, channel):
        """Handler for Button 1 (navigation)"""
        if not self.is_recording:
            # Navigate through menu when not recording
            self.menu_position = (self.menu_position + 1) % len(self.menu_items)
            self.hw.update_lcd(f"Select exercise:", self.menu_items[self.menu_position])
        else:
            # When recording, Button 1 can be used to manually increment reps
            self.rep_count += 1
            self.hw.update_lcd(f"{self.current_exercise.title()}", f"Reps: {self.rep_count} {self.form_quality}")
    
    def on_button2_press(self, channel):
        """Handler for Button 2 (selection)"""
        if not self.is_recording:
            # Select menu item
            selected = self.menu_items[self.menu_position]
            
            if selected == "Exit":
                self.stop()
            else:
                # Start exercise session
                self.current_exercise = selected
                self.start_recording()
        else:
            # Stop recording
            self.stop_recording()
    
    def start_recording(self):
        """Start exercise recording session"""
        if self.recording_thread and self.recording_thread.is_alive():
            return  # Already recording
        
        self.rep_count = 0
        self.session_data = []
        self.form_quality = "N/A"
        self.is_recording = True
        
        # Update display
        self.hw.update_lcd(f"Starting: {self.current_exercise.title()}", "Get ready...")
        self.hw.set_led(0, 0, 1)  # Blue - active
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_session)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop the current recording session"""
        self.is_recording = False
        
        # Wait for the recording thread to complete
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Save recorded data
        self.save_session_data()
        
        # Update display
        self.hw.update_lcd("Session Complete", f"Reps: {self.rep_count}")
        self.hw.set_led(0, 0, 0)  # Off
        
        # Return to menu after a delay
        time.sleep(2)
        self.hw.update_lcd(f"Select exercise:", self.menu_items[self.menu_position])
    
    def record_session(self):
        """Record and analyze exercise session in real-time"""
        print(f"Started recording session for {self.current_exercise}")
        last_update_time = 0
        
        while self.is_recording and self.is_running:
            # Get current timestamp
            timestamp = time.time()
            
            # Read MPU6050 data using the imported module methods
            mpu_data = None
            if self.mpu:
                try:
                    accel_data = self.mpu.get_accel_data()
                    gyro_data = self.mpu.get_gyro_data()
                    
                    # Calculate magnitude
                    magnitude = np.sqrt(accel_data['x']**2 + accel_data['y']**2 + accel_data['z']**2)
                    
                    # Format data to match our expected structure
                    mpu_data = {
                        'accel': accel_data,
                        'gyro': gyro_data,
                        'magnitude': magnitude
                    }
                except Exception as e:
                    print(f"Error reading MPU data: {e}")
            
            # Read and process camera frame
            pose_data = None
            if self.camera and self.pose_detector:
                ret, frame = self.camera.read()
                if ret:
                    pose_data = self.pose_detector.detect_pose(frame)
            
            # Analyze form and count repetitions
            if pose_data and mpu_data and self.current_exercise:
                # Analyze form
                self.form_quality = self.analyzer.analyze_form(
                    self.current_exercise, pose_data, mpu_data
                )
                
                # Count repetitions
                rep_increment = self.analyzer.count_repetition(mpu_data)
                if rep_increment > 0:
                    self.rep_count += rep_increment
                
                # Update LED based on form quality
                self.hw.set_led_by_form(self.form_quality)
                
                # Record data point
                data_point = {
                    "timestamp": timestamp,
                    "pose_data": pose_data,
                    "mpu_data": mpu_data,
                    "form_quality": self.form_quality,
                    "rep_count": self.rep_count
                }
                self.session_data.append(data_point)
                
                # Update LCD every 0.5 seconds to avoid flickering
                if timestamp - last_update_time >= 0.5:
                    self.hw.update_lcd(
                        f"{self.current_exercise.title()}", 
                        f"Reps: {self.rep_count} {self.form_quality[:4]}"
                    )
                    last_update_time = timestamp
            
            # Sleep to maintain reasonable frame rate
            time.sleep(0.1)
    
    def save_session_data(self):
        """Save the recorded session data to files"""
        if not self.session_data:
            return
        
        try:
            # Create a directory for the session
            session_dir = os.path.join(DATA_DIR, f"session_{self.session_id}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Save metadata
            metadata = {
                "exercise": self.current_exercise,
                "start_time": self.session_data[0]["timestamp"],
                "end_time": self.session_data[-1]["timestamp"],
                "rep_count": self.rep_count,
                "session_id": self.session_id,
                "data_points": len(self.session_data)
            }
            
            metadata_file = os.path.join(session_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save pose data
            pose_file = os.path.join(session_dir, "pose_data.json")
            pose_data_list = [{"timestamp": d["timestamp"], "pose": d["pose_data"], 
                               "form_quality": d["form_quality"], "rep_count": d["rep_count"]} 
                              for d in self.session_data if "pose_data" in d]
            with open(pose_file, 'w') as f:
                json.dump(pose_data_list, f, indent=2)
            
            # Save MPU data to CSV
            import pandas as pd
            mpu_file = os.path.join(session_dir, "mpu_data.csv")
            mpu_rows = []
            
            for d in self.session_data:
                if "mpu_data" in d and d["mpu_data"] and "accel" in d["mpu_data"] and "gyro" in d["mpu_data"]:
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
            
            print(f"Session data saved to {session_dir}")
            
        except Exception as e:
            print(f"Error saving session data: {e}")
    
    def run(self):
        """Run the main application loop"""
        self.hw.update_lcd("Fitness Coach", "Ready!")
        time.sleep(1)
        self.hw.update_lcd("Select exercise:", self.menu_items[self.menu_position])
        
        try:
            # Keep the main thread running
            while self.is_running:
                # Main loop just sleeps to reduce CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Application interrupted")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application and clean up resources"""
        print("Stopping Fitness Coach application...")
        self.is_running = False
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Clean up hardware
        self.hw.cleanup()
        
        print("Application stopped")

def main():
    """Main function to run the Fitness Coach application"""
    print("Starting Fitness Coach CLI Application")
    print(f"Using model from: {MODEL_PATH}")
    
    # Check if the model directory exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory {MODEL_PATH} not found")
        print("Please ensure the model is installed at the specified location")
        return
    
    try:
        # Create and run the Fitness Coach application
        coach = FitnessCoach()
        coach.run()
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()