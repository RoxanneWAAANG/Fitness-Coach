import requests
import json

class FitnessAPIHandler:
    def __init__(self):
        self.wger_base_url = "https://wger.de/api/v2/"
        
    def get_exercise_info(self, exercise_name):
        """Get exercise information from Wger API"""
        try:
            # Search for exercise
            search_url = f"{self.wger_base_url}exercise/search/?term={exercise_name}"
            response = requests.get(search_url)
            data = response.json()
            
            if data["count"] > 0:
                exercise_id = data["results"][0]["id"]
                
                # Get detailed information
                detail_url = f"{self.wger_base_url}exerciseinfo/{exercise_id}"
                detail_response = requests.get(detail_url)
                return detail_response.json()
            else:
                return None
                
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def get_exercise_images(self, exercise_id):
        """Get exercise images from Wger API"""
        try:
            url = f"{self.wger_base_url}exerciseimage/?exercise={exercise_id}"
            response = requests.get(url)
            return response.json()
            
        except Exception as e:
            print(f"API Error: {e}")
            return None