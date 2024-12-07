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
import socket
import json
import threading
import base64
import tempfile
import logging
from typing import Optional
import io

# Add these constants after the imports
HOST = '192.168.55.101'  # Listen on all available interfaces
PORT = 13452
BUFFER_SIZE = 4096

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
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

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
        
        analysis = f"""
        Bike Fit Analysis:
        Analyze the posture based on the bike that is being used. Whether it's mountain, road or stationary bike.
        

        Rider Details:
        - Age: {age} years
        - Height: {height} cm
        - Current Saddle Height: {saddle_height} cm

        Measured Angles:
        1. Knee Angle: {angles.get('right_knee', 'N/A')}° (Ideal: 140-150°)
        2. Hip Angle: {angles.get('right_hip', 'N/A')}° (Ideal: 45-55°)
        3. Elbow Angle: {angles.get('right_elbow', 'N/A')}° (Ideal: 150-165°)
        4. Shoulder Angle: {angles.get('shoulder_angle', 'N/A')}° (Ideal: 80-90°)
        5. Torso Angle: {angles.get('torso_angle', 'N/A')}° (Ideal: 40-45°)
        6. Knee Extension: {angles.get('knee_extension', 'N/A')}°

        Please analyze this bike fit data and provide:
        1. Detailed biomechanical analysis considering the rider's age and height
        2. Potential issues with current position
        3. Specific adjustments needed for the saddle height and other components
        4. Risk of injury assessment based on age and measurements
        5. Performance optimization suggestions
        6. Comfort recommendations based on rider's characteristics

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

    def analyze_video_for_mobile(self):
        """Modified analyze_video method that returns structured data for mobile app"""
        self.logger.info("Starting video analysis...")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        accumulated_angles = {}
        analysis_frames = []
        
        self.logger.info(f"Total frames to process: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = float((frame_count / total_frames) * 100)  # Convert to regular float
            
            # Log progress every 10% or every 30 frames, whichever comes first
            if frame_count % 30 == 0 or progress % 10 == 0:
                self.logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            results = self.model(frame, stream=True)
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    angles = self.calculate_additional_angles(keypoints)
                    
                    # Convert numpy values to regular Python floats
                    angles = {k: float(v) for k, v in angles.items()}
                    
                    # Accumulate angles for averaging
                    for angle_name, angle_value in angles.items():
                        if angle_name not in accumulated_angles:
                            accumulated_angles[angle_name] = []
                        accumulated_angles[angle_name].append(angle_value)
                    
                    # Draw skeleton and angles
                    annotated_frame = frame.copy()
                    self.draw_skeleton_and_angles(annotated_frame, keypoints, angles)
                    
                    # Save key frames (e.g., every 30th frame)
                    if frame_count % 30 == 0:
                        # Convert frame to base64 for sending to mobile
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        analysis_frames.append({
                            'frame': frame_base64,
                            'angles': angles,  # Already converted to regular floats
                            'progress': progress
                        })
        
        self.logger.info("Video processing complete. Calculating final results...")
        
        # Calculate average angles and convert to regular floats
        avg_angles = {name: float(np.nanmean(values)) for name, values in accumulated_angles.items()}
        
        self.logger.info("Getting AI recommendations...")
        # Get AI recommendations
        ai_recommendations = self.get_ai_recommendations(avg_angles)
        
        cap.release()
        
        self.logger.info("Analysis complete!")
        
        # Return structured data for mobile app
        return {
            'average_angles': avg_angles,  # Already converted to regular floats
            'recommendations': ai_recommendations,
            'key_frames': analysis_frames[:5],
            'progress': 100.0  # Use regular float
        }

class BikeFitServer:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.analyzer: Optional[BikeFitAnalyzer] = None
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the server and listen for connections"""
        try:
            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen(5)
            self.logger.info(f"Server started on {HOST}:{PORT}")

            while True:
                client_socket, address = self.server_socket.accept()
                self.logger.info(f"New connection from {address}")
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.start()

        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
        finally:
            self.server_socket.close()

    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle individual client connections"""
        try:
            # Read until we find a space character (length prefix separator)
            length_buffer = ""
            while len(length_buffer) < 10:  # Prevent infinite loop
                char = client_socket.recv(1).decode()
                if char == " ":
                    break
                length_buffer += char
                
            if not length_buffer.isdigit():
                raise ValueError(f"Invalid length prefix: {length_buffer}")
                
            msg_length = int(length_buffer)
            self.logger.info(f"Receiving message of length: {msg_length}")
            
            # Receive the full message
            data = ""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(BUFFER_SIZE, msg_length - len(data))).decode()
                if not chunk:
                    break
                data += chunk

            if not data:
                return

            # Parse the request
            request = json.loads(data)
            self.logger.info(f"Processing request: {request.get('command')}")
            
            response = self.process_request(request)
            
            # Send response length first (8 bytes, padded with spaces)
            response_json = json.dumps(response)
            length_str = f"{len(response_json):<8}"
            client_socket.send(length_str.encode())
            
            # Send the actual response
            client_socket.send(response_json.encode())
            self.logger.info("Response sent successfully")

        except Exception as e:
            self.logger.error(f"Error handling client {address}: {str(e)}")
        finally:
            client_socket.close()
            self.logger.info(f"Connection closed with {address}")

    def process_request(self, request: dict) -> dict:
        """Process incoming requests and return appropriate responses"""
        try:
            command = request.get('command')
            self.logger.info(f"Processing command: {command}")
            
            if command == 'analyze_video':
                # Handle video analysis request
                video_data = base64.b64decode(request['video_data'])
                user_details = request.get('user_details', {})
                
                self.logger.info("Received video data, saving to temporary file...")
                # Save video to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                    temp_video.write(video_data)
                    video_path = temp_video.name

                try:
                    # Create analyzer and process video
                    self.logger.info("Starting video analysis...")
                    self.analyzer = BikeFitAnalyzer(video_path, user_details)
                    analysis_result = self.analyzer.analyze_video_for_mobile()
                    
                    self.logger.info("Analysis complete, preparing response...")
                    return {
                        'status': 'success',
                        'data': {
                            'average_angles': analysis_result['average_angles'],
                            'recommendations': analysis_result['recommendations'],
                            'key_frames': analysis_result['key_frames']
                        }
                    }
                finally:
                    # Clean up temporary file
                    os.unlink(video_path)
                    self.logger.info("Temporary video file cleaned up")
                
            elif command == 'get_status':
                return {
                    'status': 'success',
                    'analyzing': self.analyzer is not None
                }
                
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown command: {command}'
                }

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    server = BikeFitServer()
    server.start()

if __name__ == "__main__":
    main()