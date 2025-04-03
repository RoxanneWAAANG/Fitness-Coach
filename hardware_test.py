# hardware_test.py
import time
import RPi.GPIO as GPIO
from hardware_controller import HardwareController

def test_hardware():
    # Initialize hardware
    hardware = HardwareController(
        btn_navigate=17,
        btn_select=27,
        red_led=16,
        green_led=20,
        blue_led=21,
        servo_pin=18
    )
    
    # Test LCD
    hardware.lcd.clear()
    hardware.lcd.write_string("Hardware Test")
    hardware.lcd.cursor_pos = (1, 0)
    hardware.lcd.write_string("Starting...")
    time.sleep(2)
    
    # Test LEDs
    hardware.lcd.clear()
    hardware.lcd.write_string("Testing LEDs")
    
    for led in [hardware.red_led, hardware.green_led, hardware.blue_led]:
        hardware.set_led(led, True)
        time.sleep(1)
        hardware.set_led(led, False)
    
    # Test servo
    hardware.lcd.clear()
    hardware.lcd.write_string("Testing Servo")
    
    for angle in [0, 90, 180, 90, 0]:
        hardware.lcd.cursor_pos = (1, 0)
        hardware.lcd.write_string(f"Angle: {angle}")
        hardware.set_servo_angle(angle)
        time.sleep(1)
    
    # Test buttons
    hardware.lcd.clear()
    hardware.lcd.write_string("Press buttons")
    hardware.lcd.cursor_pos = (1, 0)
    hardware.lcd.write_string("to test")
    
    timeout = time.time() + 10  # 10 second timeout
    while time.time() < timeout:
        if hardware.check_button_press(hardware.btn_navigate):
            hardware.lcd.clear()
            hardware.lcd.write_string("Navigate pressed")
            time.sleep(1)
            hardware.lcd.clear()
            hardware.lcd.write_string("Press buttons")
            hardware.lcd.cursor_pos = (1, 0)
            hardware.lcd.write_string("to test")
        
        if hardware.check_button_press(hardware.btn_select):
            hardware.lcd.clear()
            hardware.lcd.write_string("Select pressed")
            time.sleep(1)
            hardware.lcd.clear()
            hardware.lcd.write_string("Press buttons")
            hardware.lcd.cursor_pos = (1, 0)
            hardware.lcd.write_string("to test")
        
        time.sleep(0.1)
    
    # Test complete
    hardware.lcd.clear()
    hardware.lcd.write_string("Test complete")
    time.sleep(2)
    hardware.cleanup()

if __name__ == "__main__":
    test_hardware()