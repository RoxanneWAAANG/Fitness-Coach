#!/usr/bin/env python3
"""
Data Collection Guide - Shows how to use the data_collector.py module
directly without going through the main application.

This script provides examples for:
1. Starting a collection session
2. Recording data points
3. Ending and saving the session
"""

import time
import os
import numpy as np
from data_collector import DataCollector
import json
import datetime
import pandas as pd

def simulate_exercise_data(exercise_type, duration=30, frequency=10, user_info=None):
    """
    Simulate an exercise session with fake data for demonstration
    
    Args:
        exercise_type: Type of exercise (e.g., 'squat', 'pushup')
        duration: Duration of the simulated exercise in seconds
        frequency: How many data points to generate per second
        user_info: Dictionary containing user information (height, weight, experience)
    """
    # Initialize data collector
    collector = DataCollector(data_dir="collected_data")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(collector.data_dir, "pose_data"), exist_ok=True)
    os.makedirs(os.path.join(collector.data_dir, "mpu_data"), exist_ok=True)
    os.makedirs(os.path.join(collector.data_dir, "labeled_data"), exist_ok=True)
    
    # If no user info provided, use default values
    if user_info is None:
        user_info = {
            "height_cm": 170,
            "weight_kg": 70,
            "experience": "beginner"
        }
    
    # Start a new session
    session_id = collector.start_session(exercise_type, user_info)
    print(f"Started data collection session: {session_id}")
    
    # Initialize list to store all data points in memory
    saved_data_points = []
    
    # Simulate repetitions of exercise
    reps = 0
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Simulate accelerometer data with a sine wave pattern
        # This creates a pattern that mimics exercise motion
        phase = (elapsed % 5) / 5  # 5-second cycle for each rep
        accel_magnitude = 1.0 + 0.8 * np.sin(phase * 2 * np.pi)
        
        # Simulate a new rep every 5 seconds
        if phase < 0.1 and phase > 0:
            reps += 1
            print(f"Rep {reps} completed!")
        
        # Generate simulated pose data
        pose_data = generate_pose_data(exercise_type, phase)
        
        # Generate simulated MPU (accelerometer/gyroscope) data
        mpu_data = {
            "accel": {
                "x": 0.1 * np.sin(phase * 2 * np.pi),
                "y": 0.2 * np.cos(phase * 2 * np.pi),
                "z": accel_magnitude
            },
            "gyro": {
                "x": 0.05 * np.sin(phase * 2 * np.pi + 0.5),
                "y": 0.05 * np.cos(phase * 2 * np.pi + 0.5),
                "z": 0.02 * np.sin(phase * 4 * np.pi)
            },
            "magnitude": accel_magnitude
        }
        
        # Assign a form quality based on the phase
        # For demonstration, we'll cycle through different quality labels
        if phase < 0.3:
            form_quality = "good"
        elif phase < 0.6:
            form_quality = "moderate"
        else:
            form_quality = "poor"
        
        # Save data point to our list
        data_point = {
            "timestamp": current_time,
            "pose_data": pose_data,
            "mpu_data": mpu_data,
            "form_quality": form_quality,
            "rep_count": reps
        }
        saved_data_points.append(data_point)
        
        # Record the data point 
        collector.record_data_point(
            pose_data=pose_data,
            mpu_data=mpu_data,
            form_quality=form_quality,
            rep_count=reps,
            timestamp=current_time
        )
        
        # Sleep to maintain the desired frequency
        time.sleep(1.0 / frequency)
    
    # End the session - this will save all the data
    collector.end_session()
    
    # Make sure our session data is correctly set
    collector.session_data = saved_data_points
    collector.current_session = session_id
    collector.current_exercise = exercise_type
    
    # Manually create data files in case the automatic process failed
    
    # Save metadata (again)
    metadata = {
        "exercise": exercise_type,
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.datetime.fromtimestamp(time.time()).isoformat(),
        "user_info": user_info,
        "session_id": session_id,
        "data_points": len(saved_data_points)
    }
    
    metadata_file = os.path.join(collector.data_dir, f"session_{session_id}_meta.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save pose data
    pose_file = os.path.join(collector.data_dir, "pose_data", f"session_{session_id}_pose.json")
    pose_data_list = [{"timestamp": d["timestamp"], "pose": d["pose_data"], 
                      "form_quality": d["form_quality"], "rep_count": d["rep_count"]} 
                     for d in saved_data_points]
    with open(pose_file, 'w') as f:
        json.dump(pose_data_list, f, indent=2)
    
    # Save MPU data
    mpu_file = os.path.join(collector.data_dir, "mpu_data", f"session_{session_id}_mpu.csv")
    mpu_rows = []
    for d in saved_data_points:
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
    
    # Manually create labeled data files for training
    labeled_file = os.path.join(collector.data_dir, "labeled_data", f"session_{session_id}_labeled.csv")
    if not os.path.exists(labeled_file):
        print("Creating labeled dataset for training...")
        
        # First try using the collector method
        try:
            success = collector.create_labeled_dataset(labeled_file)
            if not success:
                raise Exception("Collector method failed to create labeled dataset")
        except Exception as e:
            print(f"Warning: {e}")
            print("Creating labeled dataset manually...")
            
            # Fallback: Create labeled data manually
            labeled_rows = []
            for d in saved_data_points:
                # Extract basic features
                features = {
                    "timestamp": d["timestamp"],
                    "label": d["form_quality"],
                    "rep_count": d["rep_count"]
                }
                
                # Add some simple pose features for key joints
                for joint in ["left_knee", "right_knee", "left_elbow", "right_elbow"]:
                    if joint in d["pose_data"]:
                        features[f"{joint}_x"] = d["pose_data"][joint]["x"]
                        features[f"{joint}_y"] = d["pose_data"][joint]["y"]
                        features[f"{joint}_conf"] = d["pose_data"][joint]["score"]
                
                # Add MPU features
                if "mpu_data" in d and "accel" in d["mpu_data"]:
                    for axis in ["x", "y", "z"]:
                        features[f"accel_{axis}"] = d["mpu_data"]["accel"][axis]
                        if "gyro" in d["mpu_data"]:
                            features[f"gyro_{axis}"] = d["mpu_data"]["gyro"][axis]
                    
                    if "magnitude" in d["mpu_data"]:
                        features["accel_magnitude"] = d["mpu_data"]["magnitude"]
                
                labeled_rows.append(features)
            
            if labeled_rows:
                pd.DataFrame(labeled_rows).to_csv(labeled_file, index=False)
    
    print(f"Data collection complete. Collected data for {reps} repetitions.")
    print(f"Data saved to: {os.path.abspath(collector.data_dir)}")
    
    # Verify data files
    files_created = []
    if os.path.exists(os.path.join(collector.data_dir, f"session_{session_id}_meta.json")):
        files_created.append("metadata")
    if os.path.exists(os.path.join(collector.data_dir, "pose_data", f"session_{session_id}_pose.json")):
        files_created.append("pose data")
    if os.path.exists(os.path.join(collector.data_dir, "mpu_data", f"session_{session_id}_mpu.csv")):
        files_created.append("MPU data")
    if os.path.exists(labeled_file):
        files_created.append("labeled data")
    
    print(f"Files created: {', '.join(files_created)}")
    
    return session_id

def generate_pose_data(exercise_type, phase):
    """Generate fake pose data based on the exercise type and phase"""
    # Base pose for a standing person (x,y coordinates and confidence score)
    base_pose = {
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
    
    # Modify the pose based on exercise type and phase
    modified_pose = base_pose.copy()
    
    if exercise_type == 'squat':
        # Simulate squat movement
        knee_y_offset = int(100 * np.sin(phase * np.pi))  # Move knees up and down
        hip_y_offset = int(50 * np.sin(phase * np.pi))    # Move hips up and down
        
        # Update joint positions
        modified_pose['left_knee']['y'] = base_pose['left_knee']['y'] - knee_y_offset
        modified_pose['right_knee']['y'] = base_pose['right_knee']['y'] - knee_y_offset
        modified_pose['left_hip']['y'] = base_pose['left_hip']['y'] - hip_y_offset
        modified_pose['right_hip']['y'] = base_pose['right_hip']['y'] - hip_y_offset
        
    elif exercise_type == 'pushup':
        # Simulate pushup movement
        elbow_y_offset = int(50 * np.sin(phase * np.pi))  # Move elbows up and down
        shoulder_y_offset = int(30 * np.sin(phase * np.pi))  # Move shoulders up and down
        
        # Update joint positions
        modified_pose['left_elbow']['y'] = base_pose['left_elbow']['y'] - elbow_y_offset
        modified_pose['right_elbow']['y'] = base_pose['right_elbow']['y'] - elbow_y_offset
        modified_pose['left_shoulder']['y'] = base_pose['left_shoulder']['y'] - shoulder_y_offset
        modified_pose['right_shoulder']['y'] = base_pose['right_shoulder']['y'] - shoulder_y_offset
        
    elif exercise_type == 'bicep_curl':
        # Simulate bicep curl movement
        wrist_y_offset = int(100 * np.sin(phase * np.pi))  # Move wrists up and down
        elbow_x_offset = int(20 * np.sin(phase * np.pi))  # Move elbows slightly
        
        # Update joint positions
        modified_pose['left_wrist']['y'] = base_pose['left_wrist']['y'] - wrist_y_offset
        modified_pose['right_wrist']['y'] = base_pose['right_wrist']['y'] - wrist_y_offset
        modified_pose['left_elbow']['x'] = base_pose['left_elbow']['x'] + elbow_x_offset
        modified_pose['right_elbow']['x'] = base_pose['right_elbow']['x'] - elbow_x_offset
    
    return modified_pose

def get_user_info():
    """
    Collect user information through an interactive menu
    
    Returns:
        Dictionary with user height, weight, and experience level
    """
    print("\n===== User Information =====")
    
    # Get height with validation
    while True:
        height_input = input("Enter your height in cm (e.g., 170): ")
        try:
            height = float(height_input)
            if 100 <= height <= 250:
                break
            else:
                print("Please enter a valid height between 100-250 cm.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get weight with validation
    while True:
        weight_input = input("Enter your weight in kg (e.g., 70): ")
        try:
            weight = float(weight_input)
            if 30 <= weight <= 200:
                break
            else:
                print("Please enter a valid weight between 30-200 kg.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get experience level
    print("\nExperience level:")
    print("1. Beginner")
    print("2. Intermediate")
    print("3. Advanced")
    
    while True:
        experience_choice = input("Select your experience level (1-3): ")
        if experience_choice == '1':
            experience = "beginner"
            break
        elif experience_choice == '2':
            experience = "intermediate"
            break
        elif experience_choice == '3':
            experience = "advanced"
            break
        else:
            print("Please select a valid option (1-3).")
    
    user_info = {
        "height_cm": height,
        "weight_kg": weight,
        "experience": experience
    }
    
    print("\nUser information recorded:")
    print(f"Height: {height} cm")
    print(f"Weight: {weight} kg")
    print(f"Experience: {experience}")
    
    return user_info

def main():
    """Main function to demonstrate data collection"""
    print("Data Collection Guide")
    print("This script demonstrates how to use the DataCollector class.")
    print("\nAvailable exercises:")
    print("1. Squat")
    print("2. Pushup")
    print("3. Bicep Curl")
    
    choice = input("\nSelect an exercise (1-3): ")
    exercise_map = {
        '1': 'squat',
        '2': 'pushup',
        '3': 'bicep_curl'
    }
    
    exercise_type = exercise_map.get(choice, 'squat')
    duration = int(input("Duration in seconds (default: 30): ") or "30")
    
    # Get user information
    user_info = get_user_info()
    
    print(f"\nSimulating {exercise_type} exercise for {duration} seconds...")
    simulate_exercise_data(exercise_type, duration, user_info=user_info)
    
    print("\nData Collection Complete!")
    print(f"""
Next steps:
1. Use the collected data to train a model:
   python model_trainer.py --exercise={exercise_type}
   
2. View the collected data:
   python data_viewer.py
""")

if __name__ == "__main__":
    main() 