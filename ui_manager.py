"""
Manages the user interface on the LCD screen
"""
import time
from datetime import datetime

class UIManager:
    def __init__(self, lcd):
        self.lcd = lcd
    
    def show_welcome(self):
        """Display welcome screen"""
        try:
            self.lcd.clear()
            self.lcd.write_string("Fitness Coach")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string("Press any button")
        except Exception as e:
            print(f"Error in show_welcome: {e}")
    
    def show_menu(self, options, hardware):
        """
        Show menu options and handle navigation with two buttons
        Returns: Index of selected option
        """
        try:
            current_option = 0
            self.lcd.clear()
            self.display_option(options[current_option], current_option, len(options))
            
            # Add timeout to prevent infinite waiting
            max_wait_time = 120  # 2 minutes timeout
            start_time = time.time()
            
            while True:
                # Check for timeout
                if time.time() - start_time > max_wait_time:
                    print("Menu selection timed out, returning default option (last one)")
                    return len(options) - 1  # Return last option (usually "Quit")
                
                try:
                    # Check for navigation button
                    if hardware.check_button_press(hardware.btn_navigate):
                        time.sleep(0.2)  # Debounce
                        current_option = (current_option + 1) % len(options)
                        self.display_option(options[current_option], current_option, len(options))
                        # Reset timeout on interaction
                        start_time = time.time()
                    
                    # Check for select button
                    if hardware.check_button_press(hardware.btn_select):
                        time.sleep(0.2)  # Debounce
                        print(f"Selected option {current_option}: {options[current_option]}")
                        return current_option
                except Exception as e:
                    print(f"Error checking buttons: {e}")
                
                # Small delay to reduce CPU usage
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in show_menu: {e}")
            # Return last option (usually "Quit" or "Back") as a safe default
            return len(options) - 1
    
    def display_option(self, option_text, current_index, total_options):
        """Display current menu option with position indicator"""
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            # Truncate if option text is too long
            display_text = f"> {option_text}"[:16]
            self.lcd.write_string(display_text)
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(f"Item {current_index+1}/{total_options}")
        except Exception as e:
            print(f"Error in display_option: {e}")
    
    def show_exercise_details(self, exercise_name, description):
        """Display exercise details"""
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(exercise_name.title()[:16])
            
            # Display description with scrolling if needed
            description_length = len(description)
            
            if description_length <= 16:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(description)
                time.sleep(3)
            else:
                # Simple scrolling text
                max_scroll = min(len(description) - 15, 20)
                for i in range(max_scroll):
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(description[i:i+16])
                    time.sleep(0.5)
        except Exception as e:
            print(f"Error in show_exercise_details: {e}")
    
    def show_message(self, title, message):
        """Display a two-line message"""
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(title[:16])
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(message[:16])
        except Exception as e:
            print(f"Error in show_message: {e}")
    
    def update_exercise_screen(self, exercise_name, rep_count, form_quality):
        """Update the exercise monitoring screen"""
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(f"Reps: {rep_count}")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(f"Form: {form_quality[:10]}")
        except Exception as e:
            print(f"Error in update_exercise_screen: {e}")
    
    def update_rep_count(self, rep_count):
        """Update just the rep count on the exercise screen"""
        try:
            self.lcd.cursor_pos = (0, 6)
            self.lcd.write_string(f"{rep_count}    ")
        except Exception as e:
            print(f"Error in update_rep_count: {e}")
    
    def show_results(self, exercise_name, rep_count, end_time):
        """Display exercise results"""
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(f"{exercise_name[:10]} done!")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(f"Reps: {rep_count}")
            time.sleep(2)
            
            # Show date/time
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string("Great workout!")
            self.lcd.cursor_pos = (1, 0)
            time_str = datetime.fromtimestamp(end_time).strftime("%H:%M %d/%m")
            self.lcd.write_string(time_str)
        except Exception as e:
            print(f"Error in show_results: {e}")
