import time
import math
import smbus2
import numpy as np

class MPU6050Handler:
    # MPU6050 Registers and Address
    ADDRESS = 0x53
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    ACCEL_YOUT_H = 0x3D
    ACCEL_ZOUT_H = 0x3F
    GYRO_XOUT_H = 0x43
    GYRO_YOUT_H = 0x45
    GYRO_ZOUT_H = 0x47
    
    def __init__(self, simulation_mode=False):
        # Data storage
        self.latest_data = None
        self.motion_history = []
        self.rep_detected = False
        
        # Calibration values
        self.accel_offset = {'x': 0, 'y': 0, 'z': 0}
        self.gyro_offset = {'x': 0, 'y': 0, 'z': 0}
        
        # Simulation mode flag
        self.simulation_mode = simulation_mode
        self.simulation_counter = 0
        
        # Try to initialize I2C bus and MPU6050
        try:
            if not simulation_mode:
                # Initialize I2C bus
                self.bus = smbus2.SMBus(1)  # Bus 1 for Raspberry Pi 3/4
                
                # Wake up the MPU6050
                self.bus.write_byte_data(self.ADDRESS, self.PWR_MGMT_1, 0)
                
                # Initialize with basic calibration
                self.calibrate()
            else:
                self.bus = None
                print("MPU6050 in simulation mode")
        except Exception as e:
            print(f"Error initializing MPU6050: {e}")
            print("Switching to simulation mode")
            self.simulation_mode = True
            self.bus = None
    
    def read_word(self, reg):
        """Read a word from the specified register"""
        if self.simulation_mode:
            return self.simulate_reading()
            
        high = self.bus.read_byte_data(self.ADDRESS, reg)
        low = self.bus.read_byte_data(self.ADDRESS, reg + 1)
        value = (high << 8) + low
        return value
    
    def read_word_2c(self, reg):
        """Convert readings to signed values"""
        val = self.read_word(reg)
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val
    
    def simulate_reading(self):
        """Generate simulated sensor data for testing"""
        # Simple simulation with sine wave to simulate movement
        self.simulation_counter += 1
        amplitude = 5000  # Adjust as needed
        frequency = 0.1   # Adjust as needed
        phase = 0
        
        # Generate sine wave with some noise
        value = amplitude * math.sin(frequency * self.simulation_counter + phase)
        value += np.random.normal(0, 500)  # Add some noise
        
        return int(value)
    
    def calibrate(self, samples=100):
        """Calibrate by taking average of multiple readings at rest"""
        if self.simulation_mode:
            print("Simulated calibration complete")
            return
            
        print("Calibrating MPU6050...")
        accel_x_sum = accel_y_sum = accel_z_sum = 0
        gyro_x_sum = gyro_y_sum = gyro_z_sum = 0
        
        try:
            for _ in range(samples):
                accel_x_sum += self.read_word_2c(self.ACCEL_XOUT_H)
                accel_y_sum += self.read_word_2c(self.ACCEL_YOUT_H)
                accel_z_sum += self.read_word_2c(self.ACCEL_ZOUT_H) - 16384  # Remove gravity
                gyro_x_sum += self.read_word_2c(self.GYRO_XOUT_H)
                gyro_y_sum += self.read_word_2c(self.GYRO_YOUT_H)
                gyro_z_sum += self.read_word_2c(self.GYRO_ZOUT_H)
                time.sleep(0.01)
            
            # Calculate offsets
            self.accel_offset['x'] = accel_x_sum / samples
            self.accel_offset['y'] = accel_y_sum / samples
            self.accel_offset['z'] = accel_z_sum / samples
            self.gyro_offset['x'] = gyro_x_sum / samples
            self.gyro_offset['y'] = gyro_y_sum / samples
            self.gyro_offset['z'] = gyro_z_sum / samples
            
            print("Calibration complete")
        except Exception as e:
            print(f"Error during calibration: {e}")
            self.simulation_mode = True
            print("Switching to simulation mode")
    
    def update_data(self):
        """Read and update sensor data"""
        try:
            if self.simulation_mode:
                # Generate simulated data
                t = time.time()
                freq = 0.5
                
                # Simulate motion pattern (repetitive movement)
                accel_x = 1.0 * math.sin(freq * t)
                accel_y = 0.5 * math.cos(freq * t)
                accel_z = 0.2 * math.sin(2 * freq * t)
                
                gyro_x = 20 * math.cos(freq * t)
                gyro_y = 15 * math.sin(freq * t)
                gyro_z = 10 * math.cos(2 * freq * t)
                
                # Add some noise
                accel_x += np.random.normal(0, 0.05)
                accel_y += np.random.normal(0, 0.05)
                accel_z += np.random.normal(0, 0.05)
                
                # Create data structure
                accel = {
                    'x': accel_x,
                    'y': accel_y,
                    'z': accel_z
                }
                
                gyro = {
                    'x': gyro_x,
                    'y': gyro_y,
                    'z': gyro_z
                }
                
                # Calculate magnitude
                accel_magnitude = math.sqrt(
                    accel_x**2 + accel_y**2 + accel_z**2
                )
                
            else:
                # Read raw data
                accel_x = self.read_word_2c(self.ACCEL_XOUT_H) - self.accel_offset['x']
                accel_y = self.read_word_2c(self.ACCEL_YOUT_H) - self.accel_offset['y']
                accel_z = self.read_word_2c(self.ACCEL_ZOUT_H) - self.accel_offset['z']
                
                gyro_x = self.read_word_2c(self.GYRO_XOUT_H) - self.gyro_offset['x']
                gyro_y = self.read_word_2c(self.GYRO_YOUT_H) - self.gyro_offset['y']
                gyro_z = self.read_word_2c(self.GYRO_ZOUT_H) - self.gyro_offset['z']
                
                # Convert to physical units
                # Accelerometer: ±2g range (16384 units per g)
                # Gyroscope: ±250 degrees per second range (131 units per deg/s)
                accel = {
                    'x': accel_x / 16384.0,
                    'y': accel_y / 16384.0,
                    'z': accel_z / 16384.0
                }
                
                gyro = {
                    'x': gyro_x / 131.0,
                    'y': gyro_y / 131.0,
                    'z': gyro_z / 131.0
                }
                
                # Calculate total acceleration (magnitude)
                accel_magnitude = math.sqrt(
                    accel['x']**2 + accel['y']**2 + accel['z']**2
                )
            
            # Store the data
            self.latest_data = {
                'accel': accel,
                'gyro': gyro,
                'magnitude': accel_magnitude,
                'timestamp': time.time(),
                'simulated': self.simulation_mode
            }
            
            # Update motion history (keep last 50 readings)
            self.motion_history.append(accel_magnitude)
            if len(self.motion_history) > 50:
                self.motion_history.pop(0)
            
            return self.latest_data
                
        except Exception as e:
            print(f"Error reading MPU6050: {e}")
            # Switch to simulation mode if real sensor fails
            if not self.simulation_mode:
                print("Switching to simulation mode")
                self.simulation_mode = True
            return self.update_data()  # Retry with simulation
    
    def read_data(self):
        """Read current sensor data"""
        self.update_data()
        return self.latest_data
    
    def detect_repetition(self, threshold=0.3):
        """
        Detect exercise repetition based on acceleration patterns
        Returns True if a repetition is detected
        """
        if len(self.motion_history) < 10:
            return False
        
        # Simple peak detection
        # Look for a significant change in acceleration followed by return to baseline
        current = self.motion_history[-1]
        peak = max(self.motion_history[-10:])
        
        # Adjust threshold for simulation mode
        if self.simulation_mode:
            detect_threshold = 0.5
            reset_threshold = 0.2
        else:
            detect_threshold = threshold
            reset_threshold = 1.2
        
        if peak > 1.5 and current < (peak - detect_threshold) and not self.rep_detected:
            self.rep_detected = True
            return True
        
        # Reset detection when acceleration returns to near resting
        if current < reset_threshold and self.rep_detected:
            self.rep_detected = False
            
        return False