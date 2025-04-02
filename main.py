"""
Main application for the Personalized Fitness Coach
This coordinates all components and manages the overall application flow
"""
import time
import threading
import RPi.GPIO as GPIO
import cv2
import numpy as np
from hardware_controller import HardwareController
from mpu6050_handler import MPU6050Handler
from camera_handler import CameraHandler
from pose_detector import PoseDetector
from exercise_analyzer import ExerciseAnalyzer
from ui_manager import UIManager
from data_collector import DataCollector
from model_trainer import ModelTrainer

# GPIO pin configuration
BTN_NAVIGATE = 17
BTN_SELECT = 27
RED_LED = 16
GREEN_LED = 20
BLUE_LED = 21
SERVO_PIN = 18

# Exercise database
EXERCISES = {
    'squat': {
        'description': 'Stand with feet shoulder-width apart, lower body as if sitting',
        'target_angles': {'knee': 90, 'hip': 45},
        'tolerance': 15,
        'keypoints': ['hip', 'knee', 'ankle'],
    },
    'pushup': {
        'description': 'Start in plank position, lower chest to ground, push back up',
        'target_angles': {'elbow': 90, 'shoulder': 30},
        'tolerance': 10,
        'keypoints': ['shoulder', 'elbow', 'wrist'],
    },
    'bicep_curl': {
        'description': 'Stand straight, curl weights up to shoulders',
        'target_angles': {'elbow': 30},
        'tolerance': 10,
        'keypoints': ['shoulder', 'elbow', 'wrist'],
    }
}

class FitnessCoach:
    def __init__(self):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Initialize components
        self.hardware = HardwareController(
            btn_navigate=BTN_NAVIGATE,
            btn_select=BTN_SELECT,
            red_led=RED_LED,
            green_led=GREEN_LED,
            blue_led=BLUE_LED,
            servo_pin=SERVO_PIN
        )
        
        self.mpu = MPU6050Handler()
        self.camera = CameraHandler()
        self.ui = UIManager(self.hardware.lcd)
        
        # Initialize analysis components
        self.pose_detector = PoseDetector()
        self.exercise_analyzer = ExerciseAnalyzer()
        
        # Application state
        self.current_exercise = None
        self.is_running = True
        self.is_monitoring = False
        self.rep_count = 0
        self.form_quality = "N/A"
        self.calibration_data = None
        
    def start(self):
        """Main application entry point"""
        try:
            self.ui.show_welcome()
            time.sleep(2)
            
            while self.is_running:
                self.main_menu()
                
        except KeyboardInterrupt:
            print("Application terminated by user")
        finally:
            self.cleanup()
    
    def main_menu(self):
        """Display and handle main menu navigation"""
        menu_options = list(EXERCISES.keys()) + ["Quit"]
        selected = self.ui.show_menu(menu_options, self.hardware)
        
        if selected == len(menu_options) - 1:  # Quit option
            self.is_running = False
        else:
            # Set current exercise based on selection
            self.current_exercise = menu_options[selected]
            self.start_exercise_workflow()
    
    def start_exercise_workflow(self):
        """Manage the complete exercise workflow"""
        if not self.current_exercise:
            return
            
        # 1. Show exercise details
        exercise_data = EXERCISES[self.current_exercise]
        self.ui.show_exercise_details(self.current_exercise, exercise_data['description'])
        
        # 2. Wait for user to get ready
        self.ui.show_message("Get ready!", "Press SELECT")
        self.hardware.wait_for_button_press(BTN_SELECT)
        
        # 3. Calibration phase
        self.calibrate()
        
        # 4. Monitoring phase
        self.monitor_exercise()
        
        # 5. Show results
        self.show_results()
    
    def calibrate(self):
        """Calibrate using initial position"""
        self.ui.show_message("Calibrating", "Hold position...")
        
        # Visual countdown with LEDs
        for i in range(3, 0, -1):
            self.ui.show_message("Calibrating", f"Hold for {i}...")
            self.hardware.blink_led(GREEN_LED, 1)
        
        # Capture calibration data
        mpu_data = self.mpu.read_data()
        camera_frame = self.camera.capture_frame()
        pose_data = self.pose_detector.detect_pose(camera_frame)
        
        self.calibration_data = {
            'mpu': mpu_data,
            'pose': pose_data
        }
        
        self.ui.show_message("Calibrated!", "Ready to start")
        time.sleep(1)
    
    def monitor_exercise(self):
        """Real-time exercise monitoring"""
        self.rep_count = 0
        self.is_monitoring = True
        
        # Get current exercise parameters
        exercise_data = EXERCISES[self.current_exercise]
        
        # Start monitoring threads
        mpu_thread = threading.Thread(target=self.mpu_monitoring_thread)
        mpu_thread.daemon = True
        mpu_thread.start()
        
        self.ui.show_message("Start Exercise", "Monitoring...")
        
        # Main monitoring loop
        start_time = time.time()
        update_interval = 0.5  # Update UI every 0.5 seconds
        last_update = start_time
        
        try:
            while self.is_monitoring:
                current_time = time.time()
                
                # Process camera input
                camera_frame = self.camera.capture_frame()
                pose_data = self.pose_detector.detect_pose(camera_frame)
                
                # Process combined data
                if pose_data and self.mpu.latest_data:
                    form_result = self.exercise_analyzer.analyze_form(
                        self.current_exercise,
                        pose_data,
                        self.mpu.latest_data,
                        exercise_data
                    )
                    
                    # Update form quality and servo position
                    self.form_quality = form_result['quality']
                    self.update_feedback(form_result)
                    
                    # Update rep count if a rep is completed
                    if form_result['rep_completed']:
                        self.rep_count += 1
                        self.ui.update_rep_count(self.rep_count)
                
                # Update UI at regular intervals
                if current_time - last_update >= update_interval:
                    self.ui.update_exercise_screen(
                        self.current_exercise,
                        self.rep_count,
                        self.form_quality
                    )
                    last_update = current_time
                
                # Check for button press to exit
                if GPIO.input(BTN_SELECT) == GPIO.HIGH:
                    time.sleep(0.2)  # Debounce
                    if GPIO.input(BTN_SELECT) == GPIO.HIGH:
                        self.is_monitoring = False
                
                # Small delay to reduce CPU usage
                time.sleep(0.05)
                
        except Exception as e:
            print(f"Error during monitoring: {e}")
        finally:
            self.is_monitoring = False
    
    def mpu_monitoring_thread(self):
        """Thread for continuous MPU6050 data reading"""
        while self.is_monitoring:
            try:
                self.mpu.update_data()
                time.sleep(0.02)  # 50Hz sampling rate
            except:
                pass
    
    def update_feedback(self, form_result):
        """Update feedback mechanisms based on form analysis"""
        quality = form_result['quality']
        
        # Update LEDs based on form quality
        if quality == 'good':
            self.hardware.set_led(GREEN_LED, True)
            self.hardware.set_led(RED_LED, False)
        elif quality == 'poor':
            self.hardware.set_led(RED_LED, True)
            self.hardware.set_led(GREEN_LED, False)
        else:  # moderate
            self.hardware.set_led(GREEN_LED, True)
            self.hardware.set_led(RED_LED, True)
        
        # Update servo position to guide movement
        # Map angle correction to servo position (0-180 degrees)
        if 'angle_correction' in form_result:
            correction = form_result['angle_correction']
            servo_angle = 90 + (correction * 0.5)  # Scale correction to servo range
            self.hardware.set_servo_angle(servo_angle)
    
    def show_results(self):
        """Display workout results"""
        self.ui.show_results(self.current_exercise, self.rep_count, time.time())
        
        # Wait for user acknowledgment
        self.hardware.wait_for_button_press(BTN_SELECT)
    
    def cleanup(self):
        """Clean up resources"""
        self.camera.release()
        self.hardware.cleanup()
        GPIO.cleanup()
        print("Resources cleaned up")

# Run the application
if __name__ == "__main__":
    app = FitnessCoach()
    app.start()
