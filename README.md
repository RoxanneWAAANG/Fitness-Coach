"""
# Personalized Fitness Coach

A Raspberry Pi-based system that analyzes workout form using camera vision and an MPU6050 motion sensor,
provides real-time feedback, and counts repetitions.

## Project Requirements

- Raspberry Pi 3 or 4
- Camera module
- MPU6050 accelerometer/gyroscope sensor
- I2C LCD display
- 2 push buttons
- Servo motor
- RGB LEDs
- Jumper wires and resistors

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fitness-coach.git
   cd fitness-coach
   ```

2. Install required dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-opencv
   pip3 install RPi.GPIO RPLCD smbus2 tflite-runtime opencv-python numpy
   ```

3. Download the MoveNet model:
   ```bash
   mkdir -p models
   # Download model - see https://www.tensorflow.org/hub/tutorials/movenet
   # or use a pre-downloaded model
   ```

## Hardware Setup

### Wiring Diagram
- See `wiring_diagram.png` for the complete circuit diagram

### Main Connections:
- LCD Display: 
  - VCC → 5V
  - GND → GND
  - SDA → GPIO2 (SDA)
  - SCL → GPIO3 (SCL)

- MPU6050:
  - VCC → 3.3V
  - GND → GND
  - SDA → GPIO2 (SDA)
  - SCL → GPIO3 (SCL)

- Buttons:
  - Button 1 → GPIO17 (with 10kΩ pull-down resistor)
  - Button 2 → GPIO27 (with 10kΩ pull-down resistor)

- RGB LEDs:
  - Red → GPIO16 (with 220Ω resistor)
  - Green → GPIO20 (with 220Ω resistor)
  - Blue → GPIO21 (with 220Ω resistor)

- Servo Motor:
  - Signal → GPIO18
  - VCC → 5V
  - GND → GND

- Camera:
  - Connect to the dedicated camera port on the Raspberry Pi

## Usage

1. Run the main application:
   ```bash
   python3 main.py
   ```

2. Follow the on-screen instructions on the LCD display:
   - Button 1: Navigate through menu options
   - Button 2: Select/confirm

3. Available exercise modes:
   - Squat
   - Pushup
   - Bicep Curl

## System Components

- **main.py**: Main application that coordinates all components
- **hardware_controller.py**: Controls LCD, buttons, LEDs, and servo
- **mpu6050_handler.py**: Interfaces with the MPU6050 sensor
- **camera_handler.py**: Manages camera operations
- **pose_detector.py**: Performs pose estimation using TensorFlow Lite
- **exercise_analyzer.py**: Analyzes exercise form and counts repetitions
- **ui_manager.py**: Manages the user interface on the LCD

## Troubleshooting

- Ensure I2C is enabled: `sudo raspi-config` → Interfacing Options → I2C → Enable
- Check camera connection: `raspistill -o test.jpg`
- Verify MPU6050 connection: `sudo i2cdetect -y 1`
- If model loading fails, provide a test model or use the demo mode

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"""
