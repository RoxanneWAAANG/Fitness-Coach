import os
import json
import time
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import cv2
from datetime import datetime
from picamera2 import Picamera2
from mpu6050 import mpu6050

class DataCollector:
    def __init__(self, data_dir="collected_data", model_path="models/movenet.tflite"):
        """Initialize data collector with MoveNet model and MPU6050 sensor"""
        self.data_dir = data_dir
        self.current_session = None
        self.current_exercise = None
        self.session_data = []
        self.camera = Picamera2()
        self.sensor = mpu6050(0x53)
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Create necessary directories
        for folder in ["pose_data", "mpu_data", "image_data", "labeled_data"]:
            os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

    def start_session(self, exercise_name, user_info=None):
        """Start a new data collection session"""
        self.current_exercise = exercise_name
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = []

        height = user_info.get("height") if user_info else None
        weight = user_info.get("weight") if user_info else None

        metadata = {
            "exercise": exercise_name,
            "start_time": datetime.now().isoformat(),
            "user_info": {
                "height": height,
                "weight": weight
            },
            "session_id": self.current_session
        }

        with open(os.path.join(self.data_dir, f"session_{self.current_session}_meta.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        return self.current_session

    def capture_image(self):
        """Capture an image and return the file path"""
        if not self.current_session:
            return None

        image_path = os.path.join(self.data_dir, "image_data", f"{self.current_session}_{int(time.time())}.jpg")
        frame = self.camera.capture_array()
        cv2.imwrite(image_path, frame)
        return image_path

    def get_mpu_data(self):
        """Retrieve MPU6050 sensor data"""
        accel = self.sensor.get_accel_data()
        gyro = self.sensor.get_gyro_data()
        return {
            "accel": accel,
            "gyro": gyro,
            "magnitude": np.linalg.norm([accel['x'], accel['y'], accel['z']])
        }

    def detect_pose(self, image_path):
        """Run MoveNet model to detect keypoints from an image"""
        image = cv2.imread(image_path)
        image = cv2.resize(image, (192, 192))
        image = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], image)
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(output_details[0]['index'])
        return keypoints.tolist()

    def record_data_point(self, form_quality, rep_count, timestamp=None):
        """Record a single data point with pose and sensor data"""
        if not self.current_session:
            return False

        timestamp = timestamp or time.time()
        mpu_data = self.get_mpu_data()
        image_path = self.capture_image()
        pose_keypoints = self.detect_pose(image_path)

        data_point = {
            "timestamp": timestamp,
            "mpu_data": mpu_data,
            "image_path": image_path,
            "pose_keypoints": pose_keypoints,
            "form_quality": form_quality,
            "rep_count": rep_count
        }

        self.session_data.append(data_point)

        if len(self.session_data) % 50 == 0:
            self.save_session_data()

        return True

    def save_session_data(self):
        """Save collected data"""
        if not self.current_session or not self.session_data:
            return False

        mpu_file = os.path.join(self.data_dir, "mpu_data", f"session_{self.current_session}_mpu.csv")
        pose_file = os.path.join(self.data_dir, "pose_data", f"session_{self.current_session}_pose.json")

        mpu_rows = [
            {
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
            for d in self.session_data
        ]

        pd.DataFrame(mpu_rows).to_csv(mpu_file, index=False)

        with open(pose_file, 'w') as f:
            json.dump([{ "timestamp": d["timestamp"], "pose": d["pose_keypoints"] } for d in self.session_data], f, indent=2)

        return True

    def end_session(self):
        """End the current session"""
        if not self.current_session:
            return False

        self.save_session_data()
        metadata_file = os.path.join(self.data_dir, f"session_{self.current_session}_meta.json")

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            metadata.update({
                "end_time": datetime.now().isoformat(),
                "data_points": len(self.session_data)
            })

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        self.current_session = None
        self.session_data = []

        return True
