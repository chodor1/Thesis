from ultralytics import YOLO
import cv2
import numpy as np
import math
from typing import List, Tuple, Dict
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import google.generativeai as genai
import os
from datetime import datetime
import customtkinter as ctk
import pyttsx3
import threading
from queue import Queue
import time

class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.speech_queue = Queue()
        self.last_spoken = {}
        self.cooldown = 3  # Seconds between repeated messages
        self.running = True
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()
    
    def _speech_worker(self):
        while self.running:
            try:
                text = self.speech_queue.get(timeout=1)
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                continue
    
    def speak(self, message: str, category: str = None):
        current_time = time.time()
        # Only speak if we haven't said this message in the last few seconds
        if category not in self.last_spoken or \
           (current_time - self.last_spoken[category]) >= self.cooldown:
            self.speech_queue.put(message)
            self.last_spoken[category] = current_time
    
    def stop(self):
        self.running = False
        self.thread.join()

class BikeFitAnalyzer:
    def __init__(self, video_path: str, user_details: dict = None):
        self.video_path = video_path
        self.user_details = user_details or {}
        self.model = YOLO(r'C:\Users\Theodore Karlyle\Documents\Thesis\yolo11s-pose.pt')
        self.keypoints_history: Dict[int, List[np.ndarray]] = {}
        self.angles_history: Dict[str, List[float]] = {
            'knee_angle': [],
            'hip_angle': [],
            'elbow_angle': [],
            'ankle_angle': [],
            'shoulder_angle': [],
            'torso_angle': [],
            'knee_extension': [],
            'hip_flexion': []
        }
        
        # Initialize Gemini API
        genai.configure(api_key='AIzaSyB_dTAXgz4vz-ALTTk7xv5Py7WZuZUlJ4s')
        self.model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        # Extended keypoint indices
        self.body_parts = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        
        self.speech_engine = SpeechEngine()
        
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        if np.any(np.isnan([p1, p2, p3])):
            return float('nan')
        
        vector1 = p1 - p2
        vector2 = p3 - p2
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_body_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate relevant angles for bike fitting"""
        angles = {}
        
        # Knee angle (ankle-knee-hip)
        angles['right_knee'] = self.calculate_angle(
            keypoints[self.body_parts['right_ankle']],
            keypoints[self.body_parts['right_knee']],
            keypoints[self.body_parts['right_hip']]
        )
        
        # Hip angle (knee-hip-shoulder)
        angles['right_hip'] = self.calculate_angle(
            keypoints[self.body_parts['right_knee']],
            keypoints[self.body_parts['right_hip']],
            keypoints[self.body_parts['right_shoulder']]
        )
        
        # Elbow angle (wrist-elbow-shoulder)
        angles['right_elbow'] = self.calculate_angle(
            keypoints[self.body_parts['right_wrist']],
            keypoints[self.body_parts['right_elbow']],
            keypoints[self.body_parts['right_shoulder']]
        )
        
        return angles

    def calculate_additional_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate additional biomechanical angles"""
        angles = self.calculate_body_angles(keypoints)  # Get basic angles first
        
        # Calculate shoulder angle (upper arm relative to torso)
        shoulder_angle = self.calculate_angle(
            keypoints[self.body_parts['right_elbow']],
            keypoints[self.body_parts['right_shoulder']],
            keypoints[self.body_parts['right_hip']]
        )
        angles['shoulder_angle'] = shoulder_angle

        # Calculate torso angle relative to vertical
        hip_point = keypoints[self.body_parts['right_hip']][:2]
        shoulder_point = keypoints[self.body_parts['right_shoulder']][:2]
        vertical_point = np.array([hip_point[0], hip_point[1] - 100])  # Point directly above hip
        torso_angle = self.calculate_angle(
            shoulder_point,
            hip_point,
            vertical_point
        )
        angles['torso_angle'] = torso_angle

        # Calculate knee extension (relative to vertical)
        knee_point = keypoints[self.body_parts['right_knee']][:2]
        ankle_point = keypoints[self.body_parts['right_ankle']][:2]
        vertical_knee = np.array([knee_point[0], knee_point[1] - 100])
        knee_extension = self.calculate_angle(
            ankle_point,
            knee_point,
            vertical_knee
        )
        angles['knee_extension'] = knee_extension

        return angles

    def analyze_posture(self, angles: Dict[str, float]) -> str:
        """Create a detailed posture analysis for Gemini API"""
        # Add user details to the analysis
        age = self.user_details.get('age', 'N/A')
        height = self.user_details.get('height', 'N/A')
        saddle_height = self.user_details.get('saddle_height', 'N/A')
        biker_history = self.user_details.get('biker_history', 'No history provided')
        
        analysis = f"""
        Bike Fit Analysis:

        Rider Details:
        - Age: {age} years
        - Height: {height} cm
        - Current Saddle Height: {saddle_height} cm
        
        Rider History:
        {biker_history}

        Measured Angles:
        1. Knee Angle: {angles.get('right_knee', 'N/A')}° (Ideal: 140-150°)
        2. Hip Angle: {angles.get('right_hip', 'N/A')}° (Ideal: 45-55°)
        3. Elbow Angle: {angles.get('right_elbow', 'N/A')}° (Ideal: 150-165°)
        4. Shoulder Angle: {angles.get('shoulder_angle', 'N/A')}° (Ideal: 80-90°)
        5. Torso Angle: {angles.get('torso_angle', 'N/A')}° (Ideal: 40-45°)
        6. Knee Extension: {angles.get('knee_extension', 'N/A')}°

        Please analyze this bike fit data and provide:
        1. Detailed biomechanical analysis considering the rider's age, height, and medical history
        2. Potential issues with current position
        3. Specific adjustments needed for the saddle height and other components
        4. Risk of injury assessment based on age, measurements, and medical history
        5. Performance optimization suggestions
        6. Comfort recommendations based on rider's characteristics and previous bike experience

        IMPORTANT: DO NOT INCLUDE THE ANGLES IN THE ANALYSIS. DO NOT MENTION AI
        """
        return analysis

    def get_ai_recommendations(self, angles: Dict[str, float]) -> str:
        """Get recommendations from Gemini API"""
        try:
            analysis = self.analyze_posture(angles)
            response = self.model_gemini.generate_content(analysis)
            return response.text
        except Exception as e:
            return f"Error getting Analysis: {str(e)}"

    def draw_skeleton_and_angles(self, frame: np.ndarray, keypoints: np.ndarray, angles: Dict[str, float]):
        """Draw skeleton and angles on the frame"""
        # Draw skeleton
        connections = [
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_shoulder', 'right_hip'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
        
        for start, end in connections:
            # Take only x,y coordinates (first 2 values) for each point
            start_point = tuple(map(int, keypoints[self.body_parts[start]][:2]))
            end_point = tuple(map(int, keypoints[self.body_parts[end]][:2]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw angles
        for angle_name, angle_value in angles.items():
            if not np.isnan(angle_value):
                if 'knee' in angle_name:
                    position = tuple(map(int, keypoints[self.body_parts['right_knee']][:2]))
                elif 'hip' in angle_name:
                    position = tuple(map(int, keypoints[self.body_parts['right_hip']][:2]))
                elif 'elbow' in angle_name:
                    position = tuple(map(int, keypoints[self.body_parts['right_elbow']][:2]))
                
                cv2.putText(frame, f'{angle_name}: {angle_value:.1f}°',
                           (position[0] + 10, position[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def get_angle_feedback(self, angles: Dict[str, float]) -> Dict[str, tuple]:
        """Generate real-time feedback based on angles with speech messages"""
        feedback = {}
        
        # Knee angle feedback (140-150° ideal)
        knee_angle = angles.get('right_knee')
        if knee_angle:
            if knee_angle < 140:
                feedback['knee'] = ("Knee angle too small - raise saddle height", 
                                  "Raise your saddle height")
            elif knee_angle > 150:
                feedback['knee'] = ("Knee angle too large - lower saddle height",
                                  "Lower your saddle height")
            else:
                feedback['knee'] = ("Knee angle optimal",
                                  "Knee position is good")
        
        # Hip angle feedback (45-55° ideal)
        hip_angle = angles.get('right_hip')
        if hip_angle:
            if hip_angle < 45:
                feedback['hip'] = ("Hip angle too closed - adjust saddle forward",
                                 "Move saddle forward")
            elif hip_angle > 55:
                feedback['hip'] = ("Hip angle too open - adjust saddle backward",
                                 "Move saddle backward")
            else:
                feedback['hip'] = ("Hip angle optimal",
                                 "Hip position is good")
        
        # Elbow angle feedback (150-165° ideal)
        elbow_angle = angles.get('right_elbow')
        if elbow_angle:
            if elbow_angle < 150:
                feedback['elbow'] = ("Arms too bent - extend reach",
                                  "Extend your reach")
            elif elbow_angle > 165:
                feedback['elbow'] = ("Arms too straight - reduce reach",
                                  "Reduce your reach")
            else:
                feedback['elbow'] = ("Elbow angle optimal",
                                  "Elbow position is good")
                
        # Torso angle feedback (40-45° ideal)
        torso_angle = angles.get('torso_angle')
        if torso_angle:
            if torso_angle < 40:
                feedback['torso'] = ("Position too aggressive - raise handlebars",
                                  "Raise handlebars")
            elif torso_angle > 45:
                feedback['torso'] = ("Position too upright - lower handlebars",
                                  "Lower handlebars")
            else:
                feedback['torso'] = ("Torso angle optimal",
                                  "Torso position is good")
        
        return feedback

    def analyze_video(self, feedback_callback=None):
        """Process video and analyze bike fit with live feedback"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_count = 0
            accumulated_angles = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                results = self.model(frame, stream=True)
                
                for result in results:
                    if result.keypoints is not None:
                        keypoints = result.keypoints.data[0].cpu().numpy()
                        angles = self.calculate_additional_angles(keypoints)
                        feedback = self.get_angle_feedback(angles)
                        
                        # Process feedback and trigger speech
                        for part, (display_msg, speech_msg) in feedback.items():
                            if "optimal" not in display_msg:  # Only speak when adjustment needed
                                self.speech_engine.speak(speech_msg, part)
                        
                        # Accumulate angles for averaging
                        for angle_name, angle_value in angles.items():
                            if angle_name not in accumulated_angles:
                                accumulated_angles[angle_name] = []
                            accumulated_angles[angle_name].append(angle_value)
                        
                        self.draw_skeleton_and_angles(frame, keypoints, angles)
                        
                        # Display real-time angles and feedback
                        y_position = 30
                        for angle_name, angle_value in angles.items():
                            cv2.putText(frame, f'{angle_name}: {angle_value:.1f}°',
                                      (10, y_position),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            y_position += 20
                        
                        # Display feedback
                        y_position = 30
                        for part, message in feedback.items():
                            color = (0, 255, 0) if "optimal" in message else (0, 165, 255)
                            cv2.putText(frame, f'{message}',
                                      (frame.shape[1] - 400, y_position),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            y_position += 20
                        
                        # Call feedback callback if provided
                        if feedback_callback:
                            feedback_callback(angles, {k: v[0] for k, v in feedback.items()})
                    
                    cv2.imshow('Bike Fit Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.speech_engine.stop()
            
            # Calculate average angles
            avg_angles = {name: np.nanmean(values) for name, values in accumulated_angles.items()}
            
            # Get final AI recommendations
            ai_recommendations = self.get_ai_recommendations(avg_angles)
            
            # Save analysis to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'bike_fit_analysis_{timestamp}.txt', 'w') as f:
                f.write("Average Angles:\n")
                for name, value in avg_angles.items():
                    f.write(f"{name}: {value:.1f}°\n")
                f.write("\nAnalysis:\n")
                f.write(ai_recommendations)
            
            return ai_recommendations
        
        except Exception as e:
            self.speech_engine.stop()
            raise e

class BikeFitUI:
    def __init__(self):
        # Set theme and color
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("Bike Fit Analyzer")
        self.root.geometry("1000x800")
        
        # Variables
        self.video_path = None
        self.analyzer = None
        self.is_analyzing = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Bike Fit Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Add user details frame
        user_details_frame = ctk.CTkFrame(self.main_frame)
        user_details_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # User details title
        details_title = ctk.CTkLabel(
            user_details_frame,
            text="Rider Details",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        details_title.pack(pady=(10, 15))
        
        # Create a frame for input fields
        inputs_frame = ctk.CTkFrame(user_details_frame)
        inputs_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        # Configure grid columns
        inputs_frame.grid_columnconfigure(0, weight=1)
        inputs_frame.grid_columnconfigure(1, weight=1)
        inputs_frame.grid_columnconfigure(2, weight=1)
        
        # Age input
        age_label = ctk.CTkLabel(inputs_frame, text="Age:", font=ctk.CTkFont(size=12))
        age_label.grid(row=0, column=0, padx=5, pady=5)
        self.age_var = tk.StringVar()
        self.age_entry = ctk.CTkEntry(
            inputs_frame,
            textvariable=self.age_var,
            width=100,
            placeholder_text="Years"
        )
        self.age_entry.grid(row=1, column=0, padx=5, pady=5)
        
        # Height input (cm)
        height_label = ctk.CTkLabel(inputs_frame, text="Height (cm):", font=ctk.CTkFont(size=12))
        height_label.grid(row=0, column=1, padx=5, pady=5)
        self.height_var = tk.StringVar()
        self.height_entry = ctk.CTkEntry(
            inputs_frame,
            textvariable=self.height_var,
            width=100,
            placeholder_text="140-200"
        )
        self.height_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Saddle height input (cm)
        saddle_label = ctk.CTkLabel(inputs_frame, text="Saddle Height (cm):", font=ctk.CTkFont(size=12))
        saddle_label.grid(row=0, column=2, padx=5, pady=5)
        self.saddle_var = tk.StringVar()
        self.saddle_entry = ctk.CTkEntry(
            inputs_frame,
            textvariable=self.saddle_var,
            width=100,
            placeholder_text="60-90"
        )
        self.saddle_entry.grid(row=1, column=2, padx=5, pady=5)
        
        # Biker History input
        history_label = ctk.CTkLabel(user_details_frame, text="Biker History:", font=ctk.CTkFont(size=12))
        history_label.pack(pady=(10, 5))
        
        self.history_text = ctk.CTkTextbox(
            user_details_frame,
            height=100,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.history_text.pack(fill="x", padx=20, pady=(0, 10))
        self.history_text.insert("0.0", "Enter medical history and previous bikes used...")
        
        # Create top frame for controls
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Upload button with modern styling
        self.upload_btn = ctk.CTkButton(
            control_frame,
            text="Upload Video",
            command=self.upload_video,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.upload_btn.pack(side="left", padx=10)
        
        # Webcam toggle
        self.webcam_var = ctk.BooleanVar()
        self.webcam_check = ctk.CTkSwitch(
            control_frame,
            text="Use Webcam",
            variable=self.webcam_var,
            command=self.toggle_webcam,
            font=ctk.CTkFont(size=14)
        )
        self.webcam_check.pack(side="left", padx=20)
        
        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            control_frame,
            text="Start Analysis",
            command=self.start_analysis,
            state="disabled",
            width=150,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.analyze_btn.pack(side="right", padx=10)
        
        # File info frame
        info_frame = ctk.CTkFrame(self.main_frame)
        info_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # File path label
        self.file_label = ctk.CTkLabel(
            info_frame,
            text="No file selected",
            font=ctk.CTkFont(size=12),
            fg_color=("gray85", "gray25"),
            corner_radius=6
        )
        self.file_label.pack(fill="x", pady=10, padx=10)
        
        # Progress bar with modern styling
        self.progress = ctk.CTkProgressBar(self.main_frame)
        self.progress.pack(fill="x", padx=20, pady=(0, 20))
        self.progress.set(0)
        
        # Create tabview for results
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True, padx=20)
        
        # Add tabs
        self.tabview.add("Analysis")
        self.tabview.add("Measurements")
        self.tabview.add("History")
        
        # Results text area with modern styling
        self.results_text = ctk.CTkTextbox(
            self.tabview.tab("Analysis"),
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Measurements tab content
        measurements_frame = ctk.CTkFrame(self.tabview.tab("Measurements"))
        measurements_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add some labels for measurements
        self.measurement_labels = {}
        measurements = ["Knee Angle", "Hip Angle", "Elbow Angle", "Shoulder Angle", "Torso Angle"]
        for i, measurement in enumerate(measurements):
            label = ctk.CTkLabel(
                measurements_frame,
                text=f"{measurement}: --°",
                font=ctk.CTkFont(size=14)
            )
            label.pack(pady=5)
            self.measurement_labels[measurement] = label
        
        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="Ready",
            font=ctk.CTkFont(size=12),
            fg_color=("gray85", "gray25"),
            corner_radius=6
        )
        self.status_bar.pack(fill="x", padx=20, pady=10)
    
    def toggle_webcam(self):
        if self.webcam_var.get():
            self.video_path = 0
            self.file_label.configure(text="Using Webcam")
            self.analyze_btn.configure(state="normal")
            self.status_bar.configure(text="Webcam mode activated")
        else:
            self.video_path = None
            self.file_label.configure(text="No file selected")
            self.analyze_btn.configure(state="disabled")
            self.status_bar.configure(text="Ready")
    
    def upload_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov"),
                ("All files", "*.*")
            ]
        )
        
        if self.video_path:
            self.file_label.configure(text=f"Selected: {os.path.basename(self.video_path)}")
            self.analyze_btn.configure(state="normal")
            self.status_bar.configure(text="Video loaded successfully")
    
    def start_analysis(self):
        if not self.is_analyzing:
            # Validate inputs
            try:
                # Get values
                age = int(self.age_var.get()) if self.age_var.get() else None
                height = float(self.height_var.get()) if self.height_var.get() else None
                saddle_height = float(self.saddle_var.get()) if self.saddle_var.get() else None
                
                # Validate ranges
                if age is not None and (age < 5 or age > 100):
                    raise ValueError("Age must be between 5 and 100 years")
                if height is not None and (height < 140 or height > 200):
                    raise ValueError("Height must be between 140 and 200 cm")
                if saddle_height is not None and (saddle_height < 60 or saddle_height > 90):
                    raise ValueError("Saddle height must be between 60 and 90 cm")
                
                user_details = {
                    'age': age,
                    'height': height,
                    'saddle_height': saddle_height,
                    'biker_history': self.history_text.get("0.0", "end").strip()
                }
                
            except ValueError as e:
                if "must be between" in str(e):
                    self.status_bar.configure(text=str(e))
                else:
                    self.status_bar.configure(text="Please enter valid numbers for all measurements")
                return
            
            self.is_analyzing = True
            self.progress.start()
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", "Analysis in progress...\n")
            self.status_bar.configure(text="Analyzing video...")
            self.analyzer = BikeFitAnalyzer(self.video_path, user_details)
            self.root.after(100, self.run_analysis)
    
    def run_analysis(self):
        try:
            def feedback_callback(angles, feedback):
                # Update measurements tab
                for name, value in angles.items():
                    if name in self.measurement_labels:
                        self.measurement_labels[name].configure(
                            text=f"{name}: {value:.1f}°"
                        )
                
                # Update analysis tab with live feedback
                feedback_text = "\n".join([f"{part}: {message}" for part, message in feedback.items()])
                self.results_text.delete("0.0", "end")
                self.results_text.insert("0.0", "Live Analysis:\n\n" + feedback_text)
                
                # Force update the UI
                self.root.update()
            
            ai_recommendations = self.analyzer.analyze_video(feedback_callback)
            
            # After analysis completes, show final AI recommendations
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", "Final Analysis:\n\n")
            self.results_text.insert("end", ai_recommendations)
            self.status_bar.configure(text="Analysis completed successfully")
            
        except Exception as e:
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", f"Error during analysis: {str(e)}")
            self.status_bar.configure(text="Error during analysis")
        
        finally:
            self.progress.stop()
            self.progress.set(0)
            self.is_analyzing = False
    
    def run(self):
        self.root.mainloop()

# Update the main function to use the UI
def main():
    app = BikeFitUI()
    app.run()

if __name__ == "__main__":
    main()
