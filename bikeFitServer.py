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

# Constants
HOST = '192.168.55.103'  # Update to match your server IP
PORT = 13452
BUFFER_SIZE = 4096

class BikeFitAnalyzer:
    def __init__(self, video_path: str, user_details: dict = None, progress_callback=None):
        self.video_path = video_path
        self.user_details = user_details or {}
        self.progress_callback = progress_callback
        self.model = YOLO(r'C:\Users\Theodore Karlyle\Documents\Thesis\yolo11s-pose.pt')
        self.keypoints_history = {}
        self.angles_history = {
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
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def send_progress(self, progress: float, message: str):
        """Send progress update through callback"""
        if self.progress_callback:
            self.progress_callback({
                'progress': float(progress),
                'message': str(message)
            })
        self.logger.info(f"Progress: {progress:.1f}% - {message}")

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        if np.any(np.isnan([p1, p2, p3])):
            return float('nan')
        
        vector1 = p1 - p2
        vector2 = p3 - p2
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return float(np.degrees(angle))

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
        
        return {k: float(v) for k, v in angles.items()}

    def calculate_additional_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate additional biomechanical angles"""
        angles = self.calculate_body_angles(keypoints)
        
        # Calculate shoulder angle (upper arm relative to torso)
        shoulder_angle = self.calculate_angle(
            keypoints[self.body_parts['right_elbow']],
            keypoints[self.body_parts['right_shoulder']],
            keypoints[self.body_parts['right_hip']]
        )
        angles['shoulder_angle'] = float(shoulder_angle)

        # Calculate torso angle relative to vertical
        hip_point = keypoints[self.body_parts['right_hip']][:2]
        shoulder_point = keypoints[self.body_parts['right_shoulder']][:2]
        vertical_point = np.array([hip_point[0], hip_point[1] - 100])
        torso_angle = self.calculate_angle(
            shoulder_point,
            hip_point,
            vertical_point
        )
        angles['torso_angle'] = float(torso_angle)

        # Calculate knee extension (relative to vertical)
        knee_point = keypoints[self.body_parts['right_knee']][:2]
        ankle_point = keypoints[self.body_parts['right_ankle']][:2]
        vertical_knee = np.array([knee_point[0], knee_point[1] - 100])
        knee_extension = self.calculate_angle(
            ankle_point,
            knee_point,
            vertical_knee
        )
        angles['knee_extension'] = float(knee_extension)

        return angles

    def analyze_posture(self, angles: Dict[str, float]) -> str:
        """Create a detailed posture analysis for Gemini API"""
        age = self.user_details.get('age', 'N/A')
        height = self.user_details.get('height', 'N/A')
        saddle_height = self.user_details.get('saddle_height', 'N/A')
        medical_history = self.user_details.get('medical_history', 'None provided')
        
        analysis = f"""
        Bike Fit Analysis:
        Analyze the posture based on the bike that is being used. Whether it's mountain, road or stationary bike.

        Rider Details:
        - Age: {age} years
        - Height: {height} cm
        - Current Saddle Height: {saddle_height} cm
        - Medical History: {medical_history}

        Measured Angles:
        1. Knee Angle: {angles.get('right_knee', 'N/A')}° (Ideal: 140-150°)
        2. Hip Angle: {angles.get('right_hip', 'N/A')}° (Ideal: 45-55°)
        3. Elbow Angle: {angles.get('right_elbow', 'N/A')}° (Ideal: 150-165°)
        4. Shoulder Angle: {angles.get('shoulder_angle', 'N/A')}° (Ideal: 80-90°)
        5. Torso Angle: {angles.get('torso_angle', 'N/A')}° (Ideal: 40-45°)
        6. Knee Extension: {angles.get('knee_extension', 'N/A')}°

        Please analyze this bike fit data and provide:
        1. Detailed biomechanical analysis considering the rider's age, height, and medical history
        2. Potential issues with current position, especially considering any medical conditions
        3. Specific adjustments needed for the saddle height and other components
        4. Risk of injury assessment based on age, medical history, and measurements
        5. Performance optimization suggestions with consideration for medical conditions
        6. Comfort recommendations based on rider's characteristics and health status
        7. Analyze video angles and display results in a mobile-friendly table with measured and ideal angles, along with posture analysis (minor/major changes or optimal). Ensure precise calculations and clear context for user clarity.

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
        connections = [
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_shoulder', 'right_hip'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
        
        for start, end in connections:
            start_point = tuple(map(int, keypoints[self.body_parts[start]][:2]))
            end_point = tuple(map(int, keypoints[self.body_parts[end]][:2]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
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
        """Modified analyze_video method that streams analysis results"""
        self.logger.info("Starting video analysis...")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        accumulated_angles = {}
        
        self.send_progress(0, "Starting video analysis...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = float((frame_count / total_frames) * 80)
            
            results = self.model(frame, stream=True)
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    angles = self.calculate_additional_angles(keypoints)
                    angles = {k: float(v) for k, v in angles.items()}
                    
                    # Accumulate angles for final average
                    for angle_name, angle_value in angles.items():
                        if angle_name not in accumulated_angles:
                            accumulated_angles[angle_name] = []
                        accumulated_angles[angle_name].append(angle_value)
                    
                    # Draw skeleton and angles on frame
                    annotated_frame = frame.copy()
                    self.draw_skeleton_and_angles(annotated_frame, keypoints, angles)
                    
                    # Convert frame to base64
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame analysis to client
                    if self.progress_callback:
                        self.progress_callback({
                            'type': 'frame_analysis',
                            'progress': progress,
                            'frame': frame_base64,
                            'angles': angles,
                            'message': f"Processing frame {frame_count}/{total_frames}"
                        })
            
            if frame_count % 10 == 0:
                self.send_progress(progress, f"Processing frame {frame_count}/{total_frames}")
        
        self.send_progress(85, "Calculating final results...")
        avg_angles = {name: float(np.nanmean(values)) for name, values in accumulated_angles.items()}
        
        self.send_progress(90, "Getting AI recommendations...")
        ai_recommendations = self.get_ai_recommendations(avg_angles)
        
        cap.release()
        
        self.send_progress(100, "Analysis complete!")
        
        return {
            'average_angles': avg_angles,
            'recommendations': ai_recommendations,
            'progress': 100.0
        }

class BikeFitServer:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.analyzer = None
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def start(self):
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

    def send_response(self, client_socket: socket.socket, response: dict):
        """Send response to client with proper formatting"""
        try:
            # Ensure all values are JSON serializable
            def convert_to_serializable(obj):
                if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple, np.ndarray)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, bytes):
                    return obj.decode('utf-8')
                return obj

            # Convert response to serializable format
            serializable_response = convert_to_serializable(response)
            response_json = json.dumps(serializable_response)
            
            # Add space after length for consistent format
            length_str = f"{len(response_json)} "
            
            # Send length first
            client_socket.sendall(length_str.encode())
            # Send JSON data
            client_socket.sendall(response_json.encode())
            
            self.logger.info(f"Sent response: {length_str}[{response_json[:100]}...]")
        except Exception as e:
            self.logger.error(f"Error sending response: {str(e)}")
            # Send error response
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            error_json = json.dumps(error_response)
            length_str = f"{len(error_json)} "
            try:
                client_socket.sendall(length_str.encode())
                client_socket.sendall(error_json.encode())
            except:
                self.logger.error("Failed to send error response")

    def handle_client(self, client_socket: socket.socket, address: tuple):
        try:
            length_buffer = ""
            while len(length_buffer) < 10:
                char = client_socket.recv(1).decode()
                if char == " ":
                    break
                length_buffer += char
                
            if not length_buffer.isdigit():
                raise ValueError(f"Invalid length prefix: {length_buffer}")
                
            msg_length = int(length_buffer)
            self.logger.info(f"Receiving message of length: {msg_length}")
            
            data = ""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(BUFFER_SIZE, msg_length - len(data))).decode()
                if not chunk:
                    break
                data += chunk

            if not data:
                return

            request = json.loads(data)
            self.logger.info(f"Processing request: {request.get('command')}")
            
            def progress_callback(progress_data):
                self.send_response(client_socket, {
                    'status': 'progress',
                    'data': progress_data
                })
            
            response = self.process_request(request, progress_callback)
            self.send_response(client_socket, response)

        except Exception as e:
            self.logger.error(f"Error handling client {address}: {str(e)}")
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            try:
                self.send_response(client_socket, error_response)
            except:
                self.logger.error("Failed to send error response")
        finally:
            client_socket.close()
            self.logger.info(f"Connection closed with {address}")

    def process_request(self, request: dict, progress_callback) -> dict:
        try:
            command = request.get('command')
            
            if command == 'analyze_video':
                video_data = base64.b64decode(request['video_data'])
                user_details = request.get('user_details', {})
                
                self.logger.info("Received video data, saving to temporary file...")
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                    temp_video.write(video_data)
                    video_path = temp_video.name

                try:
                    self.logger.info("Starting video analysis...")
                    self.analyzer = BikeFitAnalyzer(video_path, user_details, progress_callback)
                    analysis_result = self.analyzer.analyze_video_for_mobile()
                    
                    return {
                        'status': 'success',
                        'data': analysis_result
                    }
                finally:
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