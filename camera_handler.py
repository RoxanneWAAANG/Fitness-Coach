"""
Handles camera operations
"""
import cv2
import numpy as np
import time

class CameraHandler:
    def __init__(self, camera_id=0, width=640, height=480):
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Ensure camera is opened properly
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera. Check if it's connected properly.")
        
        # Warm up camera
        for _ in range(5):
            self.camera.read()
            time.sleep(0.1)
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        success, frame = self.camera.read()
        if not success:
            print("Failed to capture frame")
            return None
        
        return frame
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()