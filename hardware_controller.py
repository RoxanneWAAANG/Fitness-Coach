"""
Controls all hardware components including LCD, buttons, LEDs, and servo
"""
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

class HardwareController:
    def __init__(self, btn_navigate, btn_select, red_led, green_led, blue_led, servo_pin):
        # Set up GPIO pins
        GPIO.setmode(GPIO.BCM)
        
        # Button pins
        self.btn_navigate = btn_navigate
        self.btn_select = btn_select
        GPIO.setup(self.btn_navigate, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.btn_select, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # LED pins
        self.red_led = red_led
        self.green_led = green_led
        self.blue_led = blue_led
        GPIO.setup(self.red_led, GPIO.OUT)
        GPIO.setup(self.green_led, GPIO.OUT)
        GPIO.setup(self.blue_led, GPIO.OUT)
        
        # Servo pin setup
        self.servo_pin = servo_pin
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(self.servo_pin, 50)  # 50Hz PWM frequency
        self.servo_pwm.start(0)  # Initialize with 0% duty cycle
        
        # Initialize LCD
        # Note: I2C address may need to be adjusted for your specific LCD
        self.lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)
        self.lcd.clear()
    
    def set_led(self, led_pin, state):
        """Set LED state (True for on, False for off)"""
        GPIO.output(led_pin, GPIO.HIGH if state else GPIO.LOW)
    
    def blink_led(self, led_pin, duration, blink_rate=0.5):
        """Blink LED for specified duration with given blink rate"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(blink_rate / 2)
            GPIO.output(led_pin, GPIO.LOW)
            time.sleep(blink_rate / 2)
    
    def set_servo_angle(self, angle):
        """Set servo angle (0-180 degrees)"""
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180
            
        # Convert angle to duty cycle (0 degrees = 2.5%, 180 degrees = 12.5%)
        duty = 2.5 + (angle / 180.0 * 10.0)
        self.servo_pwm.ChangeDutyCycle(duty)
    
    def lcd_alert(self, message, flash_count=3):
        """Flash message on LCD to provide alert"""
        current_message = self.lcd.get_line(0) + self.lcd.get_line(1)
        
        for _ in range(flash_count):
            self.lcd.clear()
            time.sleep(0.3)
            self.lcd.write_string(message)
            time.sleep(0.3)
            
        # Restore original message
        self.lcd.clear()
        self.lcd.write_string(current_message)
    
    def wait_for_button_press(self, button_pin):
        """Wait for a specific button press"""
        while True:
            if GPIO.input(button_pin) == GPIO.HIGH:
                time.sleep(0.2)  # Debounce
                return True
            time.sleep(0.1)
    
    def check_button_press(self, button_pin):
        """Check if a button is currently pressed"""
        return GPIO.input(button_pin) == GPIO.HIGH
    
    def cleanup(self):
        """Clean up resources"""
        self.lcd.clear()
        self.lcd.write_string("Goodbye!")
        self.servo_pwm.stop()
        # Don't call GPIO.cleanup() here as it might be used elsewhere
