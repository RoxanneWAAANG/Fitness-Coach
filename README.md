# Personalized Fitness Coach

A Raspberry Pi-based system that collects fitness data using a camera and MPU6050 sensor, evaluates exercise form using machine learning models, and provides real-time feedback on an LCD display.

---

## Project Features

- Pose estimation with the MoveNet TFLite model
- Motion tracking using MPU6050 (accelerometer + gyroscope)
- Form quality classification using PyTorch or Random Forest models
- CLI interface for interaction and feedback
- LCD output for real-time guidance and feedback

---

## Requirements

### Hardware
- Raspberry Pi 4
- Raspberry Pi Camera
- MPU6050 sensor (I2C address: `0x53`)
- I2C LCD display (I2C address: `0x27`)
- Jumper wires, breadboard

### Software
- Python 3
- tflite-runtime
- OpenCV
- PyTorch
- scikit-learn
- pandas, numpy
- RPLCD
- Picamera2

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RoxanneWAAANG/Fitness-Coach.git
   cd Fitness-Coach
   ```

2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip libatlas-base-dev libopenblas-dev
   pip3 install numpy pandas torch torchvision opencv-python scikit-learn RPLCD smbus2 tflite-runtime
   ```

3. Enable I2C and camera:
   ```bash
   sudo raspi-config
   # Enable I2C under Interfacing Options
   # Enable Camera under Interface Options
   ```

4. Download the MoveNet model:

  - Place it in the models/ folder (e.g., models/movenet.tflite)
  - Refer to: https://www.tensorflow.org/hub/tutorials/movenet

   ```bash
   mkdir -p models
   wget -O models/movenet.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite

   ```

---

## Usage

### 1. Collect Training Data
Run the data collector to gather pose and motion data:
```bash
python3 multi_collector.py
```
Follow prompts to select an exercise (e.g., squat, pushup, bicep curl) and begin collecting labeled session data.

### 2. Train Model
Train a classification model for form quality:
```bash
python3 model_trainer.py --exercise pushup --model random_forest
# or
python3 model_trainer.py --exercise squat --model pytorch
```
Trained models will be saved in the `models/` directory.

### 3. Run Real-Time Feedback System
Launch the CLI interface for real-time analysis:
```bash
python3 fitness_coach_cli.py
```
LCD will guide you through the selected exercise and show whether your form is correct.

---

## Project Structure

- `multi_collector.py` – Captures pose + sensor data for training
- `model_trainer.py` – Trains classifiers (Random Forest / PyTorch)
- `fitness_coach_cli.py` – Main app for real-time form analysis
- `models/` – Contains TFLite pose model and trained classifiers
- `collected_data/` – Stores labeled pose and sensor datasets

---

## Troubleshooting

- **LCD not working**: Ensure correct I2C address and wiring
- **MPU6050 not detected**: Run `sudo i2cdetect -y 1` and verify it's at address `0x53`
- **Hardware error**: Test with `python utils/test_hardware.py`
- **Model loading fails**: Check file paths and naming in `fitness_coach_cli.py`

---

## License
MIT License – see the LICENSE file for full details.
