import torch
import cv2
import smbus
import time
import numpy as np
from RPLCD.i2c import CharLCD
from torchvision import transforms
from picamera2 import Picamera2
from model_trainer import FormQualityModel

# Define the I2C address for MPU6050
MPU6050_ADDR = 0x53
bus = smbus.SMBus(1)

# Initialize MPU6050
def mpu6050_init():
    bus.write_byte_data(MPU6050_ADDR, 0x6B, 0)

def read_mpu6050():
    accel_x = bus.read_word_data(MPU6050_ADDR, 0x3B)
    accel_y = bus.read_word_data(MPU6050_ADDR, 0x3D)
    accel_z = bus.read_word_data(MPU6050_ADDR, 0x3F)
    gyro_x = bus.read_word_data(MPU6050_ADDR, 0x43)
    gyro_y = bus.read_word_data(MPU6050_ADDR, 0x45)
    gyro_z = bus.read_word_data(MPU6050_ADDR, 0x47)
    return np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z], dtype=np.float32)

# Load PyTorch models
models = {}
model_paths = {
    "bicep_curl": "/home/ruoxinwang/aipi590/Fitness_Coach/Fitness-Coach/models/bicep_curl_pytorch_model.pt",
    "pushup": "/home/ruoxinwang/aipi590/Fitness_Coach/Fitness-Coach/models/pushup_pytorch_model.pt",
    "squat": "/home/ruoxinwang/aipi590/Fitness_Coach/Fitness-Coach/models/squat_pytorch_model.pt",
}

# Label mapping (as defined in model_trainer.py)
label_mapping = {0: "good", 1: "moderate", 2: "poor"}

for exercise, path in model_paths.items():
    # Initialize with the expected input size for IMU data
    model = FormQualityModel(input_size=25)
    model.load_state_dict(torch.load(path))
    model.eval()
    models[exercise] = model

# Initialize Raspberry Pi camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

def capture_frame():
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)

def display_result(exercise, form_quality):
    lcd.clear()
    lcd.write_string(f"{exercise}\nForm: {form_quality}")

def predict_exercise(selected_model=None):
    imu_data = read_mpu6050()
    frame = capture_frame()
    
    if selected_model:
        # Use only the selected model
        model = models[selected_model]
        with torch.no_grad():
            imu_tensor = torch.tensor(imu_data, dtype=torch.float32).unsqueeze(0)
            
            # Pad if needed to match expected input size
            batch_size = imu_tensor.shape[0]
            if imu_tensor.shape[1] != 25:
                padded = torch.zeros((batch_size, 25), dtype=torch.float32)
                padded[:, :imu_tensor.shape[1]] = imu_tensor
                imu_tensor = padded
            
            output = model(imu_tensor)
            # Get predicted class (0=good, 1=moderate, 2=poor)
            _, predicted_class = torch.max(output.data, 1)
            form_quality = label_mapping[predicted_class.item()]
        
        return selected_model, form_quality
    else:
        # Original behavior - try all models and pick the best
        predictions = {}
        form_qualities = {}
        
        for exercise, model in models.items():
            with torch.no_grad():
                # Reshape input to match expected size
                imu_tensor = torch.tensor(imu_data, dtype=torch.float32).unsqueeze(0)
                
                # Pad if needed to match expected input size
                batch_size = imu_tensor.shape[0]
                if imu_tensor.shape[1] != 25:
                    padded = torch.zeros((batch_size, 25), dtype=torch.float32)
                    padded[:, :imu_tensor.shape[1]] = imu_tensor
                    imu_tensor = padded
                
                # Only pass the IMU tensor to the model
                output = model(imu_tensor)
                
                # Get predicted class and confidence
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
                
                # Store the probability of "good" form (index 0) as our confidence
                predictions[exercise] = probs[0][0]
                
                # Get the predicted form quality
                _, predicted_class = torch.max(output.data, 1)
                form_qualities[exercise] = label_mapping[predicted_class.item()]
        
        # Choose the exercise with highest confidence for "good" form
        best_exercise = max(predictions, key=lambda k: predictions[k])
        form_quality = form_qualities[best_exercise]
        
        return best_exercise, form_quality

def display_menu():
    print("\n===== Fitness Coach Menu =====")
    print("Select an exercise to monitor:")
    print("1. Bicep Curl")
    print("2. Pushup")
    print("3. Squat")
    print("4. Auto-detect (evaluate all models)")
    print("5. Exit")
    print("=============================")
    return input("Enter your choice (1-5): ")

def run_fitness_coach():
    mpu6050_init()
    running = True
    selected_model = None
    
    while running:
        choice = display_menu()
        
        if choice == '1':
            selected_model = "bicep_curl"
            print(f"Selected model: {selected_model}")
        elif choice == '2':
            selected_model = "pushup"
            print(f"Selected model: {selected_model}")
        elif choice == '3':
            selected_model = "squat"
            print(f"Selected model: {selected_model}")
        elif choice == '4':
            selected_model = None
            print("Auto-detect mode enabled (using all models)")
        elif choice == '5':
            print("Exiting the Fitness Coach application...")
            running = False
            continue
        else:
            print("Invalid choice. Please try again.")
            continue
        
        print("Starting exercise monitoring (press Ctrl+C to return to menu)...")
        try:
            while True:
                exercise, form_quality = predict_exercise(selected_model)
                print(f"Detected: {exercise} - Form quality: {form_quality}")
                display_result(exercise, form_quality)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReturning to menu...")

if __name__ == "__main__":
    try:
        run_fitness_coach()
    except KeyboardInterrupt:
        print("\nExiting application...")
    finally:
        lcd.clear()
