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
        self.lcd.clear()
        self.lcd.write_string("Fitness Coach")
        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string("Press any button")
    
    def show_menu(self, options, hardware):
        """
        Show menu options and handle navigation with two buttons
        Returns: Index of selected option
        """
        current_option = 0
        self.lcd.clear()
        self.display_option(options[current_option], current_option, len(options))
        
        while True:
            # Check for navigation button
            if hardware.check_button_press(hardware.btn_navigate):
                time.sleep(0.2)  # Debounce
                current_option = (current_option + 1) % len(options)
                self.display_option(options[current_option], current_option, len(options))
            
            # Check for select button
            if hardware.check_button_press(hardware.btn_select):
                time.sleep(0.2)  # Debounce
                return current_option
            
            time.sleep(0.1)
    
    def display_option(self, option_text, current_index, total_options):
        """Display current menu option with position indicator"""
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(f"> {option_text}")
        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(f"Item {current_index+1}/{total_options}")
    
    def show_exercise_details(self, exercise_name, description):
        """Display exercise details"""
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(exercise_name.title())
        
        # Display description with scrolling if needed
        description_length = len(description)
        
        if description_length <= 16:
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(description)
            time.sleep(3)
        else:
            # Simple scrolling text
            for i in range(min(len(description) - 15, 20)):
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(description[i:i+16])
                time.sleep(0.5)
    
    def show_message(self, title, message):
        """Display a two-line message"""
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(title[:16])
        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(message[:16])
    
    def update_exercise_screen(self, exercise_name, rep_count, form_quality):
        """Update the exercise monitoring screen"""
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(f"Reps: {rep_count}")
        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(f"Form: {form_quality[:10]}")
    
    def update_rep_count(self, rep_count):
        """Update just the rep count on the exercise screen"""
        self.lcd.cursor_pos = (0, 6)
        self.lcd.write_string(f"{rep_count}    ")
    
    def show_results(self, exercise_name, rep_count, end_time):
        """Display exercise results"""
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
