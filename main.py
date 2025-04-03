"""
Main application for the Personalized Fitness Coach
This coordinates all components and manages the overall application flow
"""
import time
import threading
import sys
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
from api_handler import FitnessAPIHandler

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

# Simple terminal-based UI for test mode
class ConsoleUI:
    def show_welcome(self):
        print("\n===== Fitness Coach (TEST MODE) =====")
        print("Welcome to your personal fitness coach!")
        
    def show_menu(self, options, hardware=None):
        print("\nSelect an option:")
        for i, option in enumerate(options):
            print(f"{i+1}. {option}")
        
        while True:
            try:
                choice = int(input("Enter your choice (number): ")) - 1
                if 0 <= choice < len(options):
                    return choice
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    def show_message(self, title, message):
        print(f"\n----- {title} -----")
        print(message)
    
    def show_exercise_details(self, exercise_name, description):
        print(f"\n----- {exercise_name.upper()} -----")
        print(description)
    
    def update_exercise_screen(self, exercise_name, rep_count, form_quality):
        print(f"\rReps: {rep_count} | Form: {form_quality}", end="")
    
    def update_rep_count(self, rep_count):
        print(f"\rReps: {rep_count}", end="")
    
    def show_results(self, exercise_name, rep_count, end_time):
        print(f"\n\n----- WORKOUT COMPLETE: {exercise_name} -----")
        print(f"Total reps: {rep_count}")
        print(f"Date/Time: {time.strftime('%H:%M %d/%m', time.localtime(end_time))}")

class FitnessCoach:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        print(f"Starting in {'TEST MODE' if test_mode else 'NORMAL MODE'}")
        
        # Initialize GPIO
        if not test_mode:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
        
        # Initialize components
        try:
            if test_mode:
                print("Creating simulated hardware...")
                self.hardware = self._create_dummy_hardware()
                self.ui = ConsoleUI()
                self.mpu = self._create_dummy_mpu()
                self.camera = self._create_dummy_camera()
            else:
                print("Initializing hardware controller...")
                try:
                    self.hardware = HardwareController(
                        btn_navigate=BTN_NAVIGATE,
                        btn_select=BTN_SELECT,
                        red_led=RED_LED,
                        green_led=GREEN_LED,
                        blue_led=BLUE_LED,
                        servo_pin=SERVO_PIN
                    )
                    print("Hardware controller initialized")
                except Exception as e:
                    print(f"Error initializing hardware: {e}")
                    # Create a dummy hardware controller for testing
                    self.hardware = self._create_dummy_hardware()
                    
                print("Initializing MPU sensor...")
                self.mpu = MPU6050Handler()
                print("MPU initialized")
                
                print("Initializing camera...")
                self.camera = CameraHandler()
                print("Camera initialized")
                
                print("Initializing UI...")
                self.ui = UIManager(self.hardware.lcd)
                print("UI initialized")
            
            # Initialize analysis components
            try:
                print("Initializing pose detector...")
                self.pose_detector = PoseDetector()
                print("PoseDetector initialized successfully")
            except Exception as e:
                print(f"Error initializing PoseDetector: {e}")
                print("Using fallback mode with demo poses")
                self.pose_detector = PoseDetector()  # It will use demo data if model fails to load
            
            print("Initializing remaining components...")    
            self.exercise_analyzer = ExerciseAnalyzer()
            self.api_handler = FitnessAPIHandler()
            self.data_collector = DataCollector()
            self.model_trainer = ModelTrainer()
            print("All components initialized")
            
            # Application state
            self.current_exercise = None
            self.is_running = True
            self.is_monitoring = False
            self.is_collecting_data = False
            self.rep_count = 0
            self.form_quality = "N/A"
            self.calibration_data = None
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            # Make sure to set reasonable defaults
            self.is_running = False
            raise
    
    def _create_dummy_mpu(self):
        """Create a dummy MPU6050 for test mode"""
        class DummyMPU:
            def __init__(self):
                self.latest_data = {'accel': {'x': 0, 'y': 0, 'z': 1}, 'gyro': {'x': 0, 'y': 0, 'z': 0}}
                print("Dummy MPU6050 created")
                
            def read_data(self):
                return self.latest_data
                
            def update_data(self):
                pass
        
        return DummyMPU()
        
    def _create_dummy_camera(self):
        """Create a dummy camera for test mode"""
        class DummyCamera:
            def capture_frame(self):
                # Create a black image
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Add text
                cv2.putText(frame, "TEST MODE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return frame
                
            def release(self):
                pass
        
        return DummyCamera()
            
    def _create_dummy_hardware(self):
        """Create a dummy hardware controller for testing without physical hardware"""
        from unittest.mock import MagicMock
        
        # Create a mock hardware controller
        dummy = MagicMock()
        
        # Mock LCD
        dummy.lcd = MagicMock()
        dummy.lcd.clear = MagicMock()
        dummy.lcd.write_string = MagicMock()
        dummy.lcd.cursor_pos = (0, 0)
        
        # Mock GPIO functions
        dummy.check_button_press = MagicMock(return_value=False)
        dummy.wait_for_button_press = MagicMock(return_value=True)
        dummy.set_led = MagicMock()
        dummy.blink_led = MagicMock()
        dummy.set_servo_angle = MagicMock()
        
        # Button pins
        dummy.btn_navigate = BTN_NAVIGATE
        dummy.btn_select = BTN_SELECT
        
        print("Created dummy hardware controller for testing")
        return dummy
        
    def start(self):
        """Main application entry point"""
        try:
            print("Starting Fitness Coach application...")
            self.ui.show_welcome()
            print("Welcome message displayed")
            time.sleep(2)
            
            while self.is_running:
                print("Entering main menu...")
                self.main_menu()
                
        except KeyboardInterrupt:
            print("Application terminated by user")
        except Exception as e:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up resources...")
            self.cleanup()
    
    def main_menu(self):
        """Display and handle main menu navigation with data collection option"""
        try:
            print("Building menu options...")
            menu_options = list(EXERCISES.keys()) + ["Data Collection", "Train Model", "Quit"]
            print(f"Menu options: {menu_options}")
            
            print("Attempting to show menu...")
            try:
                selected = self.ui.show_menu(menu_options, self.hardware)
                print(f"User selected option: {selected}")
            except Exception as e:
                print(f"Error in show_menu: {e}")
                import traceback
                traceback.print_exc()
                # Default to Quit option for safety
                selected = len(menu_options) - 1
            
            if selected == len(menu_options) - 1:  # Quit option
                print("Quit selected, exiting application...")
                self.is_running = False
            elif selected == len(menu_options) - 2:  # Train Model option
                print("Train Model selected...")
                self.train_model_menu()
            elif selected == len(menu_options) - 3:  # Data Collection option
                print("Data Collection selected...")
                self.data_collection_menu()
            else:
                # Set current exercise based on selection
                self.current_exercise = menu_options[selected]
                print(f"Exercise selected: {self.current_exercise}")
                self.start_exercise_workflow()
        except Exception as e:
            print(f"Error in main_menu: {e}")
            import traceback
            traceback.print_exc()
            # Sleep a bit to avoid CPU thrashing in case of repeated errors
            time.sleep(1)

    def train_model_menu(self):
        """Menu for model training options"""
        options = ["Train Random Forest", "Train PyTorch Model", "Train U-Net Model", "Back to Main Menu"]
        self.ui.show_message("Train Model", "Select option:")
        selected = self.ui.show_menu(options, self.hardware)
        
        if selected == len(options) - 1:  # Back option
            return
            
        # Select exercise to train model for
        exercise_options = list(EXERCISES.keys()) + ["Back"]
        self.ui.show_message("Train Model", "Select exercise:")
        exercise_selected = self.ui.show_menu(exercise_options, self.hardware)
        
        if exercise_selected == len(exercise_options) - 1:  # Back option
            return
            
        exercise = exercise_options[exercise_selected]
        
        # Train model for selected exercise
        self.ui.show_message("Training", f"For {exercise}...")
        
        # This will take time - show progress on LCD
        self.ui.show_message("Training", "Please wait...")
        
        # In test mode, just simulate training
        if self.test_mode:
            print("Simulating model training...")
            time.sleep(2)
            self.ui.show_message("Training", "Success!")
            time.sleep(2)
            return
        
        # Train on a separate thread to avoid freezing the UI
        def train_thread_func():
            if selected == 0:  # Random Forest
                result = self.model_trainer.train_sklearn_model(exercise)
            elif selected == 1:  # PyTorch model
                result = self.model_trainer.train_pytorch_model(exercise)
            else:  # U-Net model
                result = self.model_trainer.train_unet_model(exercise)
                
            if result:
                self.ui.show_message("Training", "Success!")
            else:
                self.ui.show_message("Training", "Failed")
            time.sleep(2)
            
        import threading
        train_thread = threading.Thread(target=train_thread_func)
        train_thread.daemon = True
        train_thread.start()
        
        # Wait for training to complete (with button exit option)
        while train_thread.is_alive():
            if not self.test_mode and GPIO.input(self.hardware.btn_select) == GPIO.HIGH:
                self.ui.show_message("Training", "Cancelled")
                time.sleep(1)
                return
            time.sleep(0.5)
    
    def data_collection_menu(self):
        """Menu for data collection options"""
        options = list(EXERCISES.keys()) + ["Custom Exercise", "View Collected Data", "Back to Main Menu"]
        self.ui.show_message("Data Collection", "Select option:")
        selected = self.ui.show_menu(options, self.hardware)
        
        if selected == len(options) - 1:  # Back option
            return
        elif selected == len(options) - 2:  # View Collected Data
            self.view_collected_data()
            return
        elif selected == len(options) - 3:  # Custom Exercise
            self.custom_exercise_collection()
            return
            
        # Start data collection for selected exercise
        exercise = options[selected]
        self.start_data_collection(exercise)
    
    def view_collected_data(self):
        """View summary of collected data"""
        try:
            import os
            from datetime import datetime
            import json
            
            data_dir = self.data_collector.data_dir
            if not os.path.exists(data_dir):
                self.ui.show_message("No Data", "No data collected")
                time.sleep(2)
                return
                
            # Get all session metadata files
            meta_files = [f for f in os.listdir(data_dir) if f.startswith("session_") and f.endswith("_meta.json")]
            
            if not meta_files:
                self.ui.show_message("No Data", "No sessions found")
                time.sleep(2)
                return
                
            # Load metadata for all sessions
            sessions = []
            for meta_file in meta_files:
                try:
                    with open(os.path.join(data_dir, meta_file), 'r') as f:
                        meta = json.load(f)
                        sessions.append({
                            'id': meta.get('session_id', 'unknown'),
                            'exercise': meta.get('exercise', 'unknown'),
                            'start_time': meta.get('start_time', ''),
                            'data_points': meta.get('data_points', 0)
                        })
                except Exception as e:
                    print(f"Error reading metadata file {meta_file}: {e}")
            
            # Sort sessions by start time (newest first)
            sessions.sort(key=lambda x: x['start_time'], reverse=True)
            
            # Show session summary
            if self.test_mode:
                print("\n----- COLLECTED DATA SUMMARY -----")
                for i, s in enumerate(sessions[:5]):  # Show 5 most recent
                    start = datetime.fromisoformat(s['start_time']).strftime('%Y-%m-%d %H:%M')
                    print(f"{i+1}. {s['exercise']} - {start} - {s['data_points']} points")
                input("\nPress Enter to continue...")
            else:
                for i, s in enumerate(sessions):
                    start = datetime.fromisoformat(s['start_time']).strftime('%m-%d %H:%M')
                    self.ui.show_message(f"Session {i+1}/{len(sessions)}", 
                                        f"{s['exercise'][:8]} {s['data_points']}pts")
                    time.sleep(1.5)
                    
                    # Break if user presses button
                    if not self.test_mode and GPIO.input(self.hardware.btn_select) == GPIO.HIGH:
                        break
                        
                self.ui.show_message("Data Summary", f"{len(sessions)} sessions")
                time.sleep(2)
                
        except Exception as e:
            print(f"Error viewing collected data: {e}")
            import traceback
            traceback.print_exc()
            self.ui.show_message("Error", "Failed to load data")
            time.sleep(2)
    
    def custom_exercise_collection(self):
        """Collect data for a custom exercise"""
        exercise_name = "custom_exercise"
        
        if self.test_mode:
            print("\n----- CUSTOM EXERCISE DATA COLLECTION -----")
            print("Enter details for your custom exercise:")
            exercise_name = input("Exercise name: ") or "custom_exercise"
            print(f"Starting data collection for: {exercise_name}")
        else:
            # In real mode, just use a generic name with timestamp
            exercise_name = f"custom_{int(time.time() % 10000)}"
            self.ui.show_message("Custom Exercise", f"Name: {exercise_name}")
            time.sleep(2)
        
        self.start_data_collection(exercise_name, custom=True)

    def start_data_collection(self, exercise, custom=False):
        """Start data collection session for specified exercise"""
        self.current_exercise = exercise
        self.is_collecting_data = True
        
        # Add user info - in test mode can be interactive
        user_info = {}
        if self.test_mode:
            print("\nOptional: Enter user information (or press Enter to skip):")
            user_info = {
                "height_cm": input("Height in cm (optional): ") or "",
                "weight_kg": input("Weight in kg (optional): ") or "",
                "experience": input("Experience level (beginner/intermediate/advanced): ") or "beginner"
            }
        
        # Start data collection session
        session_id = self.data_collector.start_session(exercise, user_info)
        print(f"Started data collection session: {session_id}")
        
        # Show instructions
        self.ui.show_message("Data Collection", f"Started for {exercise}")
        time.sleep(1)
        
        # For a custom exercise or test mode, collect more context
        data_collection_notes = {}
        if custom or self.test_mode:
            if self.test_mode:
                print("\nPlease describe the exercise you'll perform:")
                description = input("Brief description: ") or f"Custom exercise: {exercise}"
                reps_target = input("Target number of reps: ") or "10"
                data_collection_notes = {
                    "description": description,
                    "reps_target": reps_target,
                    "focus": input("What aspect of form to focus on: ") or "general form"
                }
            else:
                data_collection_notes = {
                    "description": f"Custom exercise: {exercise}",
                    "reps_target": "10",
                    "focus": "general form"
                }
            
            # Add notes to session metadata if possible
            try:
                import os
                import json
                meta_file = os.path.join(self.data_collector.data_dir, f"session_{session_id}_meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["notes"] = data_collection_notes
                    
                    with open(meta_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"Error updating metadata: {e}")
        
        # Show instructions for the exercise
        if custom:
            self.ui.show_message("Instructions", "Perform movement")
        else:
            # Get instructions from exercise database if available
            if exercise in EXERCISES:
                self.ui.show_message("Instructions", EXERCISES[exercise]['description'][:16])
            else:
                self.ui.show_message("Instructions", "Perform exercise")
        time.sleep(2)
        
        # Calibration phase
        self.calibrate()
        
        # Monitoring and data collection phase
        self.monitor_exercise()
        
        # End data collection
        self.data_collector.end_session()
        self.is_collecting_data = False
        
        # Show completion message
        self.ui.show_message("Data Collection", "Complete!")
        time.sleep(1)
        
        # Show summary
        points_collected = len(self.data_collector.session_data) if hasattr(self.data_collector, 'session_data') else "?"
        self.ui.show_message("Data Saved", f"{points_collected} points")
        time.sleep(2)
    
    def start_exercise_workflow(self):
        """Manage the complete exercise workflow"""
        if not self.current_exercise:
            return
            
        # 1. Show exercise details
        exercise_data = EXERCISES[self.current_exercise]
        self.ui.show_exercise_details(self.current_exercise, exercise_data['description'])
        
        # 2. Wait for user to get ready
        self.ui.show_message("Get ready!", "Press SELECT")
        if not self.test_mode:
            self.hardware.wait_for_button_press(BTN_SELECT)
        else:
            input("Press Enter to continue...")
        
        # 3. Calibration phase
        self.calibrate()
        
        # 4. Monitoring phase
        self.monitor_exercise()
        
        # 5. Show results
        self.show_results()
    
    def calibrate(self):
        """Calibrate using initial position"""
        self.ui.show_message("Calibrating", "Hold position...")
        
        # Visual countdown
        for i in range(3, 0, -1):
            self.ui.show_message("Calibrating", f"Hold for {i}...")
            if not self.test_mode:
                self.hardware.blink_led(GREEN_LED, 1)
            else:
                time.sleep(1)
        
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
        
        # Start monitoring threads for MPU if not in test mode
        if not self.test_mode:
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
                if pose_data:
                    # In test mode, simulate random reps
                    if self.test_mode and random.random() < 0.1:
                        self.rep_count += 1
                        self.ui.update_rep_count(self.rep_count)
                        self.form_quality = random.choice(["good", "moderate", "poor"])
                        
                    # Normal mode processing
                    elif not self.test_mode and self.mpu.latest_data:
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

                        # Record data point if in collection mode
                        if self.is_collecting_data:
                            self.data_collector.record_data_point(
                                pose_data, 
                                self.mpu.latest_data,
                                self.form_quality,
                                self.rep_count,
                                current_time
                            )
                
                # Update UI at regular intervals
                if current_time - last_update >= update_interval:
                    self.ui.update_exercise_screen(
                        self.current_exercise,
                        self.rep_count,
                        self.form_quality
                    )
                    last_update = current_time
                
                # Check for button press or keyboard input to exit
                if self.test_mode:
                    if self.rep_count >= 10 or (hasattr(sys, 'stdin') and sys.stdin.isatty() and input == 'q'):
                        self.is_monitoring = False
                elif GPIO.input(BTN_SELECT) == GPIO.HIGH:
                    time.sleep(0.2)  # Debounce
                    if GPIO.input(BTN_SELECT) == GPIO.HIGH:
                        self.is_monitoring = False
                
                # Small delay to reduce CPU usage
                time.sleep(0.05)
                
                # For test mode, stop after 10 reps
                if self.test_mode and self.rep_count >= 10:
                    print("\nCompleted 10 reps in test mode")
                    self.is_monitoring = False
                
        except Exception as e:
            print(f"Error during monitoring: {e}")
            import traceback
            traceback.print_exc()
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
        if self.test_mode:
            return
            
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
        if not self.test_mode:
            self.hardware.wait_for_button_press(BTN_SELECT)
        else:
            input("Press Enter to continue...")
    
    def cleanup(self):
        """Clean up resources"""
        if not self.test_mode:
            self.camera.release()
            self.hardware.cleanup()
            GPIO.cleanup()
        print("Resources cleaned up")

# For test mode imports
import random

# Run the application
if __name__ == "__main__":
    # Check if test mode is enabled
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    if test_mode:
        print("Starting application in TEST MODE (no hardware required)")
    
    app = FitnessCoach(test_mode=test_mode)
    app.start()
