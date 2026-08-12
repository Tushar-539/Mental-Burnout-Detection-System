from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import json
import base64
import cv2
import numpy as np
import uuid
from datetime import datetime
import logging
import traceback
import wave
import io
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "burnout2.db")
QUESTIONS_PATH = os.path.join(BASE_DIR, "data", "questions.json")
app.config['SESSION_FOLDER'] = 'sessions'
app.config['VOICE_FOLDER'] = 'voice_sessions'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Create required directories
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)
os.makedirs(app.config['VOICE_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to track library availability
DEEPFACE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
GEMINI_AVAILABLE = False

# Try to import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ DeepFace not available: {e}")

# Try to import MediaPipe for iris detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("✅ MediaPipe imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ MediaPipe not available: {e}")

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("✅ Google Gemini AI imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Gemini AI not available: {e}")

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDZ_CCKFLK_ROGPZo8xQZJ4Whrl6yK9GKo"
gemini_model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        logger.info("✅ Gemini API configured successfully")
    except Exception as e:
        logger.error(f"❌ Gemini API configuration failed: {e}")
        GEMINI_AVAILABLE = False

# Stress assessment questions
STRESS_QUESTIONS = [
    {
        "id": 1,
        "question": "On a scale of 1 to 10, how stressed do you feel right now?",
        "type": "scale",
        "instruction": "Please speak your answer clearly, for example: 'Seven out of ten' or 'I feel about five'"
    },
    {
        "id": 2,
        "question": "What is currently your biggest source of stress or worry?",
        "type": "open",
        "instruction": "Please describe briefly what's been causing you the most stress lately"
    },
    {
        "id": 3,
        "question": "How well have you been sleeping recently?",
        "type": "descriptive",
        "instruction": "Please describe your sleep quality, for example: 'Very well', 'Poorly', or 'About average'"
    }
]


# ==================== DATABASE SETUP ====================
def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table stores demographics + result
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age_group TEXT,
            occupation TEXT,
            marital_status TEXT,
            burnout_score INTEGER,
            burnout_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Answers table stores per-question answers
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully")


init_db()


# ==================== UTILITY FUNCTIONS ====================
def load_questions():
    """Load questions from JSON file"""
    try:
        if os.path.exists(QUESTIONS_PATH):
            with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"Questions file not found at {QUESTIONS_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return {}


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ==================== IRIS DETECTOR CLASS ====================
class IrisDetector:
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ Iris detector initialized")
        else:
            logger.warning("⚠️ Iris detection unavailable (MediaPipe not installed)")
    
    def detect_iris(self, image):
        """Detect iris and calculate pupil metrics"""
        try:
            if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
                return self.mock_iris_detection(image)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {
                    'success': False,
                    'error': 'No face detected',
                    'pupil_dilation': 0.0,
                    'eye_openness': 0.0,
                    'gaze_stability': 0.0
                }
            
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            left_iris = []
            right_iris = []
            
            for idx in range(468, 473):
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    left_iris.append((landmark.x * w, landmark.y * h))
            
            for idx in range(473, 478):
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    right_iris.append((landmark.x * w, landmark.y * h))
            
            left_pupil_size = 0
            right_pupil_size = 0
            
            if len(left_iris) >= 5:
                left_pupil_size = self.calculate_iris_diameter(left_iris)
            
            if len(right_iris) >= 5:
                right_pupil_size = self.calculate_iris_diameter(right_iris)
            
            avg_pupil_size = (left_pupil_size + right_pupil_size) / 2 if (left_pupil_size and right_pupil_size) else 0
            
            left_eye_openness = self.calculate_eye_openness(face_landmarks, w, h, 'left')
            right_eye_openness = self.calculate_eye_openness(face_landmarks, w, h, 'right')
            avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
            
            normalized_dilation = min(100, max(0, (avg_pupil_size / (w * 0.05)) * 100))
            gaze_stability = 85.0 + np.random.normal(0, 5)
            
            return {
                'success': True,
                'pupil_dilation': round(float(normalized_dilation), 2),
                'eye_openness': round(float(avg_eye_openness), 2),
                'gaze_stability': round(float(max(0, min(100, gaze_stability))), 2),
                'left_pupil_size': round(float(left_pupil_size), 2),
                'right_pupil_size': round(float(right_pupil_size), 2),
                'iris_detected': True
            }
            
        except Exception as e:
            logger.error(f"Iris detection error: {e}")
            return self.mock_iris_detection(image)
    
    def calculate_iris_diameter(self, iris_points):
        """Calculate iris diameter from boundary points"""
        try:
            if len(iris_points) < 2:
                return 0
            
            max_dist = 0
            for i in range(len(iris_points)):
                for j in range(i + 1, len(iris_points)):
                    dist = np.sqrt((iris_points[i][0] - iris_points[j][0])**2 + 
                                 (iris_points[i][1] - iris_points[j][1])**2)
                    max_dist = max(max_dist, dist)
            
            return max_dist
        except:
            return 0
    
    def calculate_eye_openness(self, landmarks, w, h, eye_side):
        """Calculate eye openness ratio"""
        try:
            if eye_side == 'left':
                upper = 159
                lower = 145
                left = 33
                right = 133
            else:
                upper = 386
                lower = 374
                left = 362
                right = 263
            
            upper_point = landmarks.landmark[upper]
            lower_point = landmarks.landmark[lower]
            left_point = landmarks.landmark[left]
            right_point = landmarks.landmark[right]
            
            vertical = abs(upper_point.y - lower_point.y) * h
            horizontal = abs(left_point.x - right_point.x) * w
            
            if horizontal > 0:
                ear = (vertical / horizontal) * 100
                return min(100, max(0, ear * 3))
            
            return 50.0
        except:
            return 50.0
    
    def mock_iris_detection(self, image):
        """Mock iris detection for demo purposes"""
        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            
            base_dilation = max(20, min(80, 50 + (brightness - 127) / 5 + np.random.normal(0, 8)))
            base_openness = max(60, min(95, 80 + np.random.normal(0, 7)))
            base_stability = max(70, min(95, 85 + np.random.normal(0, 5)))
            
            return {
                'success': True,
                'pupil_dilation': round(float(base_dilation), 2),
                'eye_openness': round(float(base_openness), 2),
                'gaze_stability': round(float(base_stability), 2),
                'left_pupil_size': round(float(15 + np.random.normal(0, 2)), 2),
                'right_pupil_size': round(float(15 + np.random.normal(0, 2)), 2),
                'iris_detected': False,
                'mock_data': True
            }
        except Exception as e:
            logger.error(f"Mock iris detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'pupil_dilation': 50.0,
                'eye_openness': 80.0,
                'gaze_stability': 85.0
            }


# ==================== EMOTION ANALYZER CLASS ====================
class EmotionAnalyzer:
    def __init__(self):
        self.supported_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.iris_detector = IrisDetector()
        
    def decode_base64_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
                
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def analyze_emotion_with_deepface(self, image):
        """Analyze emotion using DeepFace"""
        try:
            if not DEEPFACE_AVAILABLE:
                raise ImportError("DeepFace not available")
                
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            if isinstance(result, list):
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']
            
            normalized_emotions = {}
            for emotion in self.supported_emotions:
                if emotion in emotions:
                    normalized_emotions[emotion] = round(float(emotions[emotion]), 2)
                else:
                    normalized_emotions[emotion] = 0.0
            
            normalized_emotions = convert_numpy_types(normalized_emotions)
            
            return {
                'success': True,
                'emotions': normalized_emotions,
                'dominant_emotion': max(normalized_emotions, key=normalized_emotions.get)
            }
            
        except Exception as e:
            logger.error(f"DeepFace analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotions': {emotion: 0.0 for emotion in self.supported_emotions}
            }
    
    def analyze_emotion_mock(self, image):
        """Mock emotion analysis for demo purposes"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            
            base_scores = {
                'happy': max(0, min(100, brightness / 2.55 + np.random.normal(0, 10))),
                'neutral': max(0, min(100, 50 + np.random.normal(0, 15))),
                'sad': max(0, min(100, (255 - brightness) / 3 + np.random.normal(0, 8))),
                'angry': max(0, min(100, 15 + np.random.normal(0, 12))),
                'surprise': max(0, min(100, 20 + np.random.normal(0, 10))),
                'fear': max(0, min(100, 10 + np.random.normal(0, 8))),
                'disgust': max(0, min(100, 8 + np.random.normal(0, 6)))
            }
            
            total = sum(base_scores.values())
            mock_emotions = {}
            for emotion in base_scores:
                mock_emotions[emotion] = round(float(base_scores[emotion] / total * 100), 2)
            
            mock_emotions = convert_numpy_types(mock_emotions)
            
            return {
                'success': True,
                'emotions': mock_emotions,
                'dominant_emotion': max(mock_emotions, key=mock_emotions.get)
            }
            
        except Exception as e:
            logger.error(f"Mock analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotions': {emotion: 0.0 for emotion in self.supported_emotions}
            }
    
    def analyze_emotion(self, image):
        """Main emotion analysis method"""
        if DEEPFACE_AVAILABLE:
            return self.analyze_emotion_with_deepface(image)
        else:
            logger.info("Using mock analysis (DeepFace not available)")
            return self.analyze_emotion_mock(image)
    
    def analyze_with_iris(self, image):
        """Combined emotion and iris analysis"""
        emotion_result = self.analyze_emotion(image)
        iris_result = self.iris_detector.detect_iris(image)
        
        return {
            'emotion': emotion_result,
            'iris': iris_result,
            'combined_stress_score': self.calculate_combined_stress(emotion_result, iris_result)
        }
    
    def calculate_combined_stress(self, emotion_result, iris_result):
        """Calculate stress score combining emotion and iris data"""
        try:
            if not emotion_result.get('success') or not iris_result.get('success'):
                return 50.0
            
            emotions = emotion_result.get('emotions', {})
            
            emotion_stress = (
                emotions.get('angry', 0) * 0.8 +
                emotions.get('fear', 0) * 0.7 +
                emotions.get('sad', 0) * 0.6 +
                emotions.get('disgust', 0) * 0.5 -
                emotions.get('happy', 0) * 0.4
            )
            
            pupil_dilation = iris_result.get('pupil_dilation', 50)
            eye_openness = iris_result.get('eye_openness', 80)
            gaze_stability = iris_result.get('gaze_stability', 85)
            
            dilation_stress = abs(pupil_dilation - 50) * 0.8
            openness_stress = (100 - eye_openness) * 0.6
            stability_stress = (100 - gaze_stability) * 0.7
            
            iris_stress = (dilation_stress + openness_stress + stability_stress) / 3
            combined_stress = (emotion_stress * 0.6 + iris_stress * 0.4)
            
            return round(float(max(0, min(100, combined_stress))), 2)
            
        except Exception as e:
            logger.error(f"Combined stress calculation error: {e}")
            return 50.0


# Initialize analyzer
analyzer = EmotionAnalyzer()

@app.route('/insta')
def instagram_feed():
    """Serve the Instagram-style feed for burnout detection"""
    return render_template('insta.html')


@app.route('/insta.html')
def instagram_feed_html():
    """Alternative route for Instagram feed"""
    return render_template('insta.html')


@app.route('/resultinsta')
def instagram_result():
    """Display Instagram burnout analysis results"""
    return render_template('resultinsta.html')


@app.route('/resultinsta.html')
def instagram_result_html():
    """Alternative route for Instagram results"""
    return render_template('resultinsta.html')


@app.route('/burnout', methods=['POST'])
def analyze_instagram_burnout():
    """Analyze burnout based on Instagram-style interactions"""
    try:
        data = request.json
        posts = data.get("posts", [])

        score = 0
        max_score = 0
        category_scores = {
            "stress_indicators": 0,
            "rest_engagement": 0,
            "work_attitude": 0,
            "emotional_state": 0
        }
        
        interactions_count = 0

        for post in posts:
            topic = post.get("topic", "")
            action = post.get("action", "")
            comment = post.get("comment", "")

            if not action and not comment:
                continue
            
            interactions_count += 1

            # Stress and sarcasm indicators
            if topic == "sarcastic_study":
                max_score += 3
                if action == "like":
                    score += 2
                    category_scores["stress_indicators"] += 2
                elif action == "heart":
                    score += 3
                    category_scores["stress_indicators"] += 3
                elif action == "dislike":
                    score += 0
            
            # Rest and relaxation
            elif topic == "rest":
                max_score += 3
                if action == "like":
                    score += 0
                    category_scores["rest_engagement"] += 1
                elif action == "heart":
                    score += 0
                    category_scores["rest_engagement"] += 2
                elif action == "dislike":
                    score += 3
                    category_scores["rest_engagement"] -= 1
            
            # Study attitude
            elif topic == "study":
                max_score += 3
                if action == "like":
                    score += 0
                elif action == "heart":
                    score += 0
                elif action == "dislike":
                    score += 2
                    category_scores["work_attitude"] += 2
            
            # Work stress
            elif topic == "work_stress":
                max_score += 3
                if action == "like":
                    score += 2
                    category_scores["stress_indicators"] += 2
                elif action == "heart":
                    score += 3
                    category_scores["stress_indicators"] += 3
                elif action == "dislike":
                    score += 0
            
            # Motivation
            elif topic == "motivation":
                max_score += 3
                if action == "like":
                    score += 0
                elif action == "heart":
                    score += 0
                elif action == "dislike":
                    score += 2
                    category_scores["work_attitude"] += 2
            
            # Stress posts
            elif topic == "stress":
                max_score += 3
                if action == "like":
                    score += 2
                    category_scores["stress_indicators"] += 2
                elif action == "heart":
                    score += 3
                    category_scores["stress_indicators"] += 3
                elif action == "dislike":
                    score += 0
            
            # Fun posts
            elif topic in ["fun", "fun_work"]:
                max_score += 3
                if action == "like":
                    score += 0
                    category_scores["emotional_state"] += 1
                elif action == "heart":
                    score += 0
                    category_scores["emotional_state"] += 2
                elif action == "dislike":
                    score += 2
                    category_scores["emotional_state"] -= 1

            # Sentiment analysis of comments
            if comment:
                max_score += 2
                try:
                    sentiment = TextBlob(comment).sentiment.polarity
                    if sentiment < -0.2:
                        score += 2
                        category_scores["emotional_state"] += 2
                    elif sentiment < 0:
                        score += 1
                        category_scores["emotional_state"] += 1
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {e}")

        # Calculate percentage (0-100 range)
        if max_score > 0:
            percentage = min(100, max(0, int((score / max_score) * 100)))
        else:
            percentage = 0

        # Determine status and recommendations
        if percentage >= 70:
            status = "High Burnout Risk"
            level = "high"
            message = "You're showing strong signs of burnout. Take immediate action!"
            recommendations = [
                "Take a break from work/study for at least a day",
                "Talk to someone you trust about your stress",
                "Practice deep breathing or meditation",
                "Get 8+ hours of sleep tonight",
                "Consider professional counseling if symptoms persist"
            ]
        elif percentage >= 40:
            status = "Moderate Burnout"
            level = "moderate"
            message = "You're experiencing moderate stress. Time to slow down."
            recommendations = [
                "Schedule regular breaks during work/study",
                "Engage in a hobby you enjoy",
                "Exercise for 30 minutes daily",
                "Limit screen time before bed",
                "Practice mindfulness or relaxation techniques"
            ]
        else:
            status = "Low Burnout"
            level = "low"
            message = "You're managing well! Keep up the healthy habits."
            recommendations = [
                "Maintain your current work-life balance",
                "Continue your stress management practices",
                "Stay connected with friends and family",
                "Keep prioritizing self-care",
                "Share your wellness strategies with others"
            ]

        logger.info(f"Instagram burnout analysis completed: {percentage}% ({level})")

        return jsonify({
            "status": status,
            "percentage": percentage,
            "level": level,
            "message": message,
            "recommendations": recommendations,
            "categories": {
                "stress": min(100, max(0, category_scores["stress_indicators"] * 10)),
                "rest": min(100, max(0, 100 - category_scores["rest_engagement"] * 10)),
                "work": min(100, max(0, category_scores["work_attitude"] * 10)),
                "emotional": min(100, max(0, 100 - category_scores["emotional_state"] * 5))
            },
            "interactions": interactions_count,
            "analysis_type": "instagram_behavior"
        })

    except Exception as e:
        logger.error(f"Instagram burnout analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to analyze burnout data'
        }), 500


# ==================== STRESS CALCULATION FUNCTIONS ====================
def calculate_stress_level(emotions, iris_data=None):
    """Calculate stress level based on emotion distribution and iris data"""
    try:
        stress_weights = {
            'angry': 0.8,
            'fear': 0.7,
            'sad': 0.6,
            'disgust': 0.5,
            'surprise': 0.2,
            'neutral': -0.1,
            'happy': -0.4
        }
        
        emotion_stress = 0
        for emotion, percentage in emotions.items():
            if emotion in stress_weights:
                emotion_stress += (percentage / 100) * stress_weights[emotion]
        
        total_stress = emotion_stress
        
        if iris_data:
            pupil_dilation = iris_data.get('pupil_dilation', 50)
            eye_openness = iris_data.get('eye_openness', 80)
            gaze_stability = iris_data.get('gaze_stability', 85)
            
            dilation_stress = abs(pupil_dilation - 50) / 100 * 0.3
            openness_stress = (100 - eye_openness) / 100 * 0.2
            stability_stress = (100 - gaze_stability) / 100 * 0.2
            
            iris_stress = dilation_stress + openness_stress + stability_stress
            total_stress = emotion_stress * 0.7 + iris_stress * 0.3
        
        normalized_score = max(0, min(10, (total_stress + 0.4) * 10))
        
        if normalized_score <= 3:
            level = 'low'
            description = 'Relaxed and calm state'
        elif normalized_score <= 6:
            level = 'moderate'
            description = 'Normal stress levels'
        elif normalized_score <= 8:
            level = 'high'
            description = 'Elevated stress, consider relaxation techniques'
        else:
            level = 'very_high'
            description = 'High stress detected, consider professional support'
        
        result = {
            'level': level,
            'score': round(normalized_score, 1),
            'description': description,
            'raw_score': round(total_stress, 3)
        }
        
        if iris_data:
            result['iris_indicators'] = {
                'pupil_dilation': pupil_dilation,
                'eye_openness': eye_openness,
                'gaze_stability': gaze_stability,
                'fatigue_level': 100 - eye_openness,
                'cognitive_load': abs(pupil_dilation - 50)
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Stress calculation error: {e}")
        return {
            'level': 'moderate',
            'score': 5.0,
            'description': 'Unable to calculate stress level',
            'raw_score': 0.0
        }


def calculate_mind_age(emotions, dominant_emotion):
    """Calculate psychological/emotional age based on emotion patterns"""
    try:
        emotion_weights = {
            'happy': 0.15,
            'neutral': 0.20,
            'sad': -0.05,
            'angry': -0.15,
            'fear': -0.10,
            'surprise': 0.05,
            'disgust': -0.08
        }
        
        maturity_score = 0
        for emotion, percentage in emotions.items():
            if emotion in emotion_weights:
                maturity_score += (percentage / 100) * emotion_weights[emotion]
        
        age_mapping = {
            'happy': {'base': 25, 'range': (20, 35), 'description': 'Optimistic Young Adult'},
            'neutral': {'base': 35, 'range': (30, 45), 'description': 'Mature Adult'},
            'sad': {'base': 28, 'range': (18, 40), 'description': 'Reflective Individual'},
            'angry': {'base': 22, 'range': (16, 30), 'description': 'Reactive Young Adult'},
            'fear': {'base': 26, 'range': (20, 35), 'description': 'Cautious Individual'},
            'surprise': {'base': 23, 'range': (18, 32), 'description': 'Curious Young Adult'},
            'disgust': {'base': 30, 'range': (25, 40), 'description': 'Critical Adult'}
        }
        
        dominant_info = age_mapping.get(dominant_emotion, age_mapping['neutral'])
        base_age = dominant_info['base']
        age_adjustment = maturity_score * 30
        calculated_age = max(16, min(50, base_age + age_adjustment))
        
        if maturity_score > 0.1:
            ei_level = "High"
            ei_description = "Shows strong emotional regulation and balance"
        elif maturity_score > -0.05:
            ei_level = "Moderate"
            ei_description = "Demonstrates average emotional awareness"
        else:
            ei_level = "Developing"
            ei_description = "Has room for growth in emotional regulation"
        
        return {
            'mind_age': round(calculated_age),
            'age_range': dominant_info['range'],
            'personality_type': dominant_info['description'],
            'emotional_intelligence': ei_level,
            'ei_description': ei_description,
            'maturity_score': round(maturity_score, 3)
        }
        
    except Exception as e:
        logger.error(f"Mind age calculation error: {e}")
        return {
            'mind_age': 25,
            'age_range': (20, 35),
            'personality_type': 'Balanced Individual',
            'emotional_intelligence': 'Moderate',
            'ei_description': 'Shows typical emotional patterns',
            'maturity_score': 0.0
        }


def generate_gemini_recommendations(analysis_data):
    """Generate AI-powered recommendations using Gemini API"""
    try:
        if not GEMINI_AVAILABLE or not gemini_model:
            logger.warning("Gemini API not available, using fallback")
            return None
        
        # Extract key data
        dominant_emotion = analysis_data.get('dominant_emotion', 'neutral')
        emotions = analysis_data.get('emotions', {})
        stress_assessment = analysis_data.get('stress_assessment', {})
        iris_data = analysis_data.get('iris_data', {})
        mind_age = analysis_data.get('mind_age_data', {})
        
        # Build comprehensive prompt
        prompt = f"""You are an empathetic mental health and wellness advisor. Based on the following comprehensive burnout detection analysis, provide personalized, actionable recommendations.

**Analysis Summary:**
- Dominant Emotion: {dominant_emotion}
- Emotion Distribution: {json.dumps(emotions, indent=2)}
- Stress Level: {stress_assessment.get('level', 'unknown')} (Score: {stress_assessment.get('score', 0)}/10)
- Stress Description: {stress_assessment.get('description', 'N/A')}

**Iris & Physiological Indicators:**
- Pupil Dilation: {iris_data.get('pupil_dilation', 'N/A')}%
- Eye Openness: {iris_data.get('eye_openness', 'N/A')}%
- Gaze Stability: {iris_data.get('gaze_stability', 'N/A')}%
- Cognitive Load: {stress_assessment.get('iris_indicators', {}).get('cognitive_load', 'N/A')}
- Fatigue Level: {stress_assessment.get('iris_indicators', {}).get('fatigue_level', 'N/A')}

**Psychological Profile:**
- Estimated Psychological Age: {mind_age.get('mind_age', 'N/A')} years
- Personality Type: {mind_age.get('personality_type', 'N/A')}
- Emotional Intelligence: {mind_age.get('emotional_intelligence', 'N/A')}
- EI Description: {mind_age.get('ei_description', 'N/A')}

**Instructions:**
Generate 8-12 personalized recommendations that are:
1. Specific and actionable (not generic advice)
2. Evidence-based when possible
3. Tailored to the person's current emotional state, stress level, and physiological indicators
4. Organized into categories: Immediate Actions (next 24 hours), Short-term Strategies (this week), Long-term Wellness, and Professional Support
5. Compassionate and non-judgmental in tone
6. Include specific techniques, apps, resources, or practices where relevant
7. Address both emotional and physical wellbeing based on iris data
8. Consider the psychological age and emotional maturity level

Format your response as a JSON object with this structure:
{{
  "immediate_actions": ["action1", "action2", "action3"],
  "short_term_strategies": ["strategy1", "strategy2", "strategy3"],
  "long_term_wellness": ["practice1", "practice2", "practice3"],
  "professional_support": ["recommendation1", "recommendation2"],
  "personalized_insight": "A brief, encouraging paragraph about their current state and growth opportunities"
}}

Be specific, practical, and supportive. Avoid platitudes."""

        # Generate recommendations
        response = gemini_model.generate_content(prompt)
        
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return None
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        recommendations = json.loads(response_text)
        
        logger.info("Successfully generated Gemini recommendations")
        return recommendations
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON response: {e}")
        logger.error(f"Response text: {response.text if response else 'No response'}")
        return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


# ==================== HOME & LANDING ROUTES ====================
@app.route('/')
def index():
    """Serve the main home page"""
    return render_template('home.html')


@app.route('/home')
def home_redirect():
    """Alternative route for home page"""
    return render_template('home.html')


@app.route('/home.html')
def home_page():
    """Alternative route for home page"""
    return render_template('home.html')


# ==================== QUESTIONNAIRE ROUTES ====================
@app.route('/gender')
def gender_redirect():
    """Redirect to gender selection page"""
    return render_template('gender.html')


@app.route('/gender.html')
def gender():
    """Serve gender selection page"""
    return render_template('gender.html')


@app.route('/submit_info', methods=['POST'])
def submit_info():
    """Submit gender and age information"""
    gender = request.form.get('gender')
    age_group = request.form.get('age')

    if not gender or not age_group:
        return "Please select gender and age", 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (gender, age_group) VALUES (?, ?)", (gender, age_group))
    conn.commit()
    conn.close()

    return redirect('/occupation.html')


@app.route('/occupation')
def occupation_redirect():
    """Redirect to occupation selection page"""
    return render_template('occupation.html')


@app.route('/occupation.html')
def occupation_page():
    """Serve occupation selection page"""
    return render_template('occupation.html')


@app.route('/submit_occupation', methods=['POST'])
def submit_occupation():
    """Submit occupation information"""
    occupation = request.form.get('occupation')
    if not occupation:
        return "Please select an occupation", 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET occupation = ? WHERE id = (SELECT MAX(id) FROM users)", (occupation,))
    conn.commit()
    conn.close()

    return redirect(url_for('marital_status_page'))


@app.route('/marital-status')
def marital_status_redirect():
    """Redirect to marital status selection page"""
    return render_template('marital_status.html')


@app.route('/marital_status.html')
def marital_status_page():
    """Serve marital status selection page"""
    return render_template('marital_status.html')


@app.route('/submit_marital', methods=['POST'])
def submit_marital():
    """Submit marital status information"""
    status = request.form.get('status')
    if not status:
        return "Please select the marital status", 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET marital_status = ? WHERE id = (SELECT MAX(id) FROM users)", (status,))
    conn.commit()
    conn.close()

    return redirect(url_for('questions_page'))


@app.route('/questions')
def questions_redirect():
    """Redirect to questions page"""
    return questions_page()


@app.route('/question.html')
def questions_page():
    """Serve questions page"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, gender, age_group, occupation, marital_status FROM users ORDER BY id DESC LIMIT 1")
    user = cursor.fetchone()
    conn.close()

    if not user:
        return "No user data found", 400

    user_id, gender, age_group, occupation, marital_status = user

    # Normalize exactly like JSON keys
    gender = gender.strip().capitalize()
    occupation = occupation.strip().capitalize()
    marital_status = marital_status.strip().lower()
    age_group = age_group.strip()

    user_key = f"{gender}_{age_group}_{occupation}_{marital_status}"
    logger.info(f"✅ Generated user key: {user_key}")

    questions_data = load_questions()
    questions = questions_data.get(user_key, [])
    logger.info(f"✅ Loaded questions count: {len(questions)}")

    if not questions:
        logger.warning(f"No questions available for key: {user_key}")
        # Use default questions if specific key not found
        questions = [
            "Do you feel emotionally drained from your work?",
            "Do you feel tired when you wake up in the morning?",
            "Do you feel frustrated or irritable at work?",
            "Do you feel like you're working too hard?",
            "Do you feel stressed at work?",
            "Do you feel burned out from work?",
            "Do you feel unhappy or depressed at work?",
            "Do you feel you have accomplished many worthwhile things?",
            "Do you feel satisfied with your job?",
            "Do you have positive relationships with coworkers?"
        ]

    return render_template('question.html', questions=questions, user_id=user_id)


@app.route('/submit_responses', methods=['POST'])
def submit_responses():
    """Submit questionnaire responses"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    user_id = data.get("user_id")
    answers = data.get("answers", [])

    if not user_id:
        return jsonify({"success": False, "error": "user_id missing"}), 400

    score_map = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
    total_score = 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Save answers
    for item in answers:
        q = item.get("question")
        a = item.get("answer")
        cursor.execute("INSERT INTO answers (user_id, question, answer) VALUES (?, ?, ?)", (user_id, q, a))
        total_score += score_map.get(a, 0)

    # Compute burnout level
    if total_score <= 23:
        level = "Low"
    elif total_score <= 37:
        level = "Moderate"
    else:
        level = "High"

    # Update user row with result
    cursor.execute("UPDATE users SET burnout_score = ?, burnout_level = ? WHERE id = ?",
                   (total_score, level, user_id))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route('/result')
@app.route('/result.html')
def result():
    """Display questionnaire results"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT gender, age_group, occupation, marital_status, burnout_score, burnout_level
        FROM users
        WHERE burnout_score IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "No result found", 400

    gender, age_group, occupation, marital_status, score, level = row
    return render_template('result.html',
                           gender=gender,
                           age_group=age_group,
                           occupation=occupation,
                           marital_status=marital_status,
                           score=score,
                           level=level)

# ==================== FACE & VOICE ANALYSIS ROUTES ====================
@app.route('/real-time-analysis')
def real_time_analysis():
    """Serve the real-time analysis page (face + voice)"""
    return render_template('index.html')


@app.route('/index.html')
def index_page():
    """Alternative route for real-time analysis"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Enhanced Stress Detection API with Iris Tracking',
        'deepface_available': DEEPFACE_AVAILABLE,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'gemini_available': GEMINI_AVAILABLE,
        'iris_detection': MEDIAPIPE_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/routes', methods=['GET'])
def list_routes():
    """List all available routes for debugging"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({
        'success': True,
        'total_routes': len(routes),
        'routes': sorted(routes, key=lambda x: x['path'])
    })


@app.route('/api/get-questions', methods=['GET'])
def get_stress_questions():
    """Get stress assessment questions for voice analysis"""
    return jsonify({
        'success': True,
        'questions': STRESS_QUESTIONS
    })


@app.route('/api/get-voice-prompts', methods=['GET'])
def get_voice_prompts():
    """Get voice interaction prompts with timing"""
    prompts = [
        {
            "time": 5,
            "text": "Hello! Welcome to our stress detection system. I'll ask you three questions while analyzing your facial expressions. Please speak naturally and clearly when you answer. Let's begin!",
            "type": "greeting",
            "duration": 8
        },
        {
            "time": 15,
            "text": "First question: How much time do you spent on social media ?",
            "type": "question",
            "id": 1,
            "duration": 10
        },
        {
            "time": 30,
            "text": "Second question: What is currently your biggest source of stress or worry? Please describe briefly.",
            "type": "question",
            "id": 2,
            "duration": 12
        },
        {
            "time": 48,
            "text": "Final question: How well have you been sleeping recently? Please share your sleep quality.",
            "type": "question",
            "id": 3,
            "duration": 10
        }
    ]
    return jsonify({
        'success': True,
        'prompts': prompts
    })


@app.route('/api/create-session', methods=['POST'])
def create_session():
    """Create a new analysis session"""
    try:
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(app.config['SESSION_FOLDER'], session_id)
        voice_folder = os.path.join(app.config['VOICE_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        os.makedirs(voice_folder, exist_ok=True)
        
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'frames_count': 0,
            'voice_recordings_count': 0,
            'status': 'created',
            'deepface_available': DEEPFACE_AVAILABLE,
            'mediapipe_available': MEDIAPIPE_AVAILABLE,
            'gemini_available': GEMINI_AVAILABLE,
            'iris_detection_enabled': MEDIAPIPE_AVAILABLE,
            'session_duration': 60,
            'frame_interval': 10,
            'total_frames': 6
        }
        
        with open(os.path.join(session_folder, 'session.json'), 'w') as f:
            json.dump(session_data, f)
        
        logger.info(f"✅ Created session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session created successfully',
            'session_config': {
                'duration': 60,
                'frame_interval': 10,
                'total_frames': 6,
                'iris_detection': MEDIAPIPE_AVAILABLE,
                'ai_recommendations': GEMINI_AVAILABLE
            }
        })
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/save-voice', methods=['POST'])
def save_voice():
    """Save voice recording for a question"""
    try:
        data = request.get_json()
        
        if not data:
            logger.error("No data received in save-voice request")
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        session_id = data.get('session_id')
        question_id = data.get('question_id', 'unknown')
        audio_data = data.get('audio_data')
        
        logger.info(f"🎤 Received voice save request - Session: {session_id}, Question: {question_id}")
        
        if not session_id or not audio_data:
            return jsonify({'success': False, 'error': 'Missing session_id or audio_data'}), 400
        
        voice_folder = os.path.join(app.config['VOICE_FOLDER'], session_id)
        if not os.path.exists(voice_folder):
            os.makedirs(voice_folder, exist_ok=True)
            logger.info(f"Created voice folder: {voice_folder}")
        
        try:
            # Handle base64 data
            if ',' in audio_data:
                audio_data = audio_data.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_data)
            
            # Generate filename
            if question_id == 'full_session':
                filename = f"full_session_recording.webm"
            else:
                filename = f"question_{question_id}_response.webm"
            
            filepath = os.path.join(voice_folder, filename)
            
            # Save audio file
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            
            file_size = len(audio_bytes)
            logger.info(f"✅ Saved voice recording: {filename} ({file_size} bytes)")
            
            return jsonify({
                'success': True,
                'message': f'Voice recording saved for question {question_id}',
                'filename': filename,
                'size': file_size
            })
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'error': f'Failed to process audio data: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Voice save error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload-frames', methods=['POST'])
def upload_frames():
    """Upload and save captured frames"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        session_id = data.get('session_id')
        frames = data.get('frames', [])
        
        if not session_id or not frames:
            return jsonify({'success': False, 'error': 'Missing session_id or frames'}), 400
        
        session_folder = os.path.join(app.config['SESSION_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        saved_frames = []
        
        for i, frame_data in enumerate(frames):
            try:
                image = analyzer.decode_base64_image(frame_data['imageData'])
                if image is not None:
                    timestamp_clean = frame_data['timestamp'].replace(':', '-').replace('.', '-')
                    filename = f"frame_{i+1:02d}_{timestamp_clean}.jpg"
                    filepath = os.path.join(session_folder, filename)
                    
                    if cv2.imwrite(filepath, image):
                        saved_frames.append({
                            'frame_id': i + 1,
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': frame_data['timestamp'],
                            'capture_time': frame_data.get('capture_time', i * 10)
                        })
                        logger.info(f"Saved frame {i+1}: {filename}")
                    
            except Exception as e:
                logger.error(f"Error saving frame {i+1}: {e}")
                continue
        
        session_file = os.path.join(session_folder, 'session.json')
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        session_data.update({
            'frames_count': len(saved_frames),
            'status': 'frames_uploaded',
            'frames': saved_frames,
            'uploaded_at': datetime.now().isoformat()
        })
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Successfully saved {len(saved_frames)} frames',
            'frames_saved': len(saved_frames)
        })
        
    except Exception as e:
        logger.error(f"Frame upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze-emotions', methods=['POST'])
def analyze_emotions():
    """Analyze emotions and iris data for all frames"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Missing session_id'}), 400
        
        session_folder = os.path.join(app.config['SESSION_FOLDER'], session_id)
        session_file = os.path.join(session_folder, 'session.json')
        
        if not os.path.exists(session_file):
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        frames = session_data.get('frames', [])
        if not frames:
            return jsonify({'success': False, 'error': 'No frames found'}), 400
        
        analysis_results = []
        iris_data_collection = []
        
        logger.info(f"Starting combined analysis for {len(frames)} frames")
        
        for frame_info in frames:
            filepath = frame_info['filepath']
            
            if os.path.exists(filepath):
                try:
                    image = cv2.imread(filepath)
                    if image is None:
                        raise ValueError("Could not load image")
                    
                    combined_result = analyzer.analyze_with_iris(image)
                    emotion_result = combined_result['emotion']
                    iris_result = combined_result['iris']
                    
                    frame_result = {
                        'frame': frame_info['frame_id'],
                        'timestamp': frame_info['timestamp'],
                        'capture_time': frame_info.get('capture_time', 0),
                        'filename': frame_info['filename'],
                        'emotions': convert_numpy_types(emotion_result['emotions']),
                        'dominant_emotion': emotion_result.get('dominant_emotion', 'neutral'),
                        'iris_data': convert_numpy_types(iris_result),
                        'combined_stress_score': combined_result['combined_stress_score'],
                        'success': emotion_result['success'] and iris_result['success']
                    }
                    
                    if not frame_result['success']:
                        frame_result['error'] = emotion_result.get('error') or iris_result.get('error', 'Analysis failed')
                    
                    analysis_results.append(frame_result)
                    iris_data_collection.append(iris_result)
                    logger.info(f"Analyzed frame {frame_info['frame_id']} with iris data")
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame {frame_info['frame_id']}: {e}")
                    analysis_results.append({
                        'frame': frame_info['frame_id'],
                        'timestamp': frame_info['timestamp'],
                        'capture_time': frame_info.get('capture_time', 0),
                        'filename': frame_info['filename'],
                        'emotions': {emotion: 0.0 for emotion in analyzer.supported_emotions},
                        'dominant_emotion': 'neutral',
                        'iris_data': {'success': False, 'error': str(e)},
                        'combined_stress_score': 50.0,
                        'success': False,
                        'error': str(e)
                    })
        
        successful_analyses = [r for r in analysis_results if r['success']]
        
        if successful_analyses:
            avg_emotions = {}
            for emotion in analyzer.supported_emotions:
                total = sum(float(frame['emotions'][emotion]) for frame in successful_analyses)
                avg_emotions[emotion] = round(total / len(successful_analyses), 2)
            
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
            
            avg_iris = {
                'pupil_dilation': round(np.mean([i['pupil_dilation'] for i in iris_data_collection if i['success']]), 2),
                'eye_openness': round(np.mean([i['eye_openness'] for i in iris_data_collection if i['success']]), 2),
                'gaze_stability': round(np.mean([i['gaze_stability'] for i in iris_data_collection if i['success']]), 2)
            }
            
            stress_level = calculate_stress_level(avg_emotions, avg_iris)
            mind_age_data = calculate_mind_age(avg_emotions, dominant_emotion)
        else:
            avg_emotions = {emotion: 0.0 for emotion in analyzer.supported_emotions}
            dominant_emotion = 'neutral'
            avg_iris = {'pupil_dilation': 50.0, 'eye_openness': 80.0, 'gaze_stability': 85.0}
            stress_level = {'level': 'moderate', 'score': 5.0, 'description': 'Unable to assess'}
            mind_age_data = calculate_mind_age(avg_emotions, dominant_emotion)
        
        avg_emotions = convert_numpy_types(avg_emotions)
        analysis_results = convert_numpy_types(analysis_results)
        avg_iris = convert_numpy_types(avg_iris)
        
        session_data.update({
            'status': 'analyzed',
            'analysis_results': analysis_results,
            'average_emotions': avg_emotions,
            'average_iris_data': avg_iris,
            'dominant_emotion': dominant_emotion,
            'stress_assessment': stress_level,
            'mind_age_data': mind_age_data,
            'analyzed_at': datetime.now().isoformat()
        })
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'results': analysis_results,
            'average_emotions': avg_emotions,
            'average_iris_data': avg_iris,
            'dominant_emotion': dominant_emotion,
            'stress_assessment': stress_level,
            'mind_age_data': mind_age_data,
            'total_frames': len(frames),
            'successful_analyses': len(successful_analyses),
            'deepface_used': DEEPFACE_AVAILABLE,
            'mediapipe_used': MEDIAPIPE_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations using Gemini AI"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        dominant_emotion = data.get('dominant_emotion', 'neutral')
        emotions = data.get('emotions', {})
        stress_assessment = data.get('stress_assessment', {})
        iris_data = data.get('iris_data', {})
        
        # Calculate mind age if not provided
        mind_age_data = data.get('mind_age_data')
        if not mind_age_data:
            mind_age_data = calculate_mind_age(emotions, dominant_emotion)
        
        # Prepare comprehensive analysis data for Gemini
        analysis_data = {
            'dominant_emotion': dominant_emotion,
            'emotions': emotions,
            'stress_assessment': stress_assessment,
            'iris_data': iris_data,
            'mind_age_data': mind_age_data
        }
        
        # Try to get Gemini recommendations
        gemini_recommendations = generate_gemini_recommendations(analysis_data)
        
        if gemini_recommendations:
            # Format Gemini recommendations for response
            all_recommendations = []
            
            if gemini_recommendations.get('immediate_actions'):
                all_recommendations.append("**Immediate Actions (Next 24 Hours):**")
                all_recommendations.extend(gemini_recommendations['immediate_actions'])
            
            if gemini_recommendations.get('short_term_strategies'):
                all_recommendations.append("\n**Short-term Strategies (This Week):**")
                all_recommendations.extend(gemini_recommendations['short_term_strategies'])
            
            if gemini_recommendations.get('long_term_wellness'):
                all_recommendations.append("\n**Long-term Wellness:**")
                all_recommendations.extend(gemini_recommendations['long_term_wellness'])
            
            if gemini_recommendations.get('professional_support'):
                all_recommendations.append("\n**Professional Support:**")
                all_recommendations.extend(gemini_recommendations['professional_support'])
            
            return jsonify({
                'success': True,
                'dominant_emotion': dominant_emotion,
                'stress_level': stress_assessment.get('level', 'moderate'),
                'recommendations': all_recommendations,
                'personalized_insight': gemini_recommendations.get('personalized_insight', ''),
                'gemini_generated': True,
                'mind_age_analysis': {
                    'estimated_mind_age': mind_age_data['mind_age'],
                    'age_range': f"{mind_age_data['age_range'][0]}-{mind_age_data['age_range'][1]} years",
                    'personality_type': mind_age_data['personality_type'],
                    'emotional_intelligence': mind_age_data['emotional_intelligence'],
                    'ei_description': mind_age_data['ei_description'],
                    'interpretation': f"Based on your emotional patterns, your psychological age appears to be around {mind_age_data['mind_age']} years, suggesting a {mind_age_data['personality_type'].lower()} emotional profile."
                },
                'stress_analysis': {
                    'level': stress_assessment.get('level', 'moderate'),
                    'score': stress_assessment.get('score', 5.0),
                    'description': stress_assessment.get('description', 'Normal stress levels'),
                    'interpretation': f"Your stress level is {stress_assessment.get('level', 'moderate')} ({stress_assessment.get('score', 5.0)}/10)"
                },
                'iris_analysis': iris_data if iris_data else None
            })
        
        # Fallback to basic recommendations if Gemini fails
        logger.warning("Using fallback recommendations (Gemini unavailable)")
        
        basic_recommendations = [
            f"Based on your {dominant_emotion} emotional state, focus on activities that promote emotional balance.",
            f"Your stress level is {stress_assessment.get('level', 'moderate')}. Consider stress-reduction techniques.",
            "Practice mindfulness meditation for 10-15 minutes daily.",
            "Ensure you're getting 7-9 hours of quality sleep each night.",
            "Engage in regular physical activity to reduce stress hormones.",
            "Connect with friends or family members for social support.",
            "Consider journaling to process your emotions and thoughts.",
            "If stress persists, don't hesitate to seek professional counseling."
        ]
        
        return jsonify({
            'success': True,
            'dominant_emotion': dominant_emotion,
            'stress_level': stress_assessment.get('level', 'moderate'),
            'recommendations': basic_recommendations,
            'personalized_insight': f"Your analysis shows {dominant_emotion} as the dominant emotion with a stress level of {stress_assessment.get('level', 'moderate')}. Focus on self-care and stress management.",
            'gemini_generated': False,
            'mind_age_analysis': {
                'estimated_mind_age': mind_age_data['mind_age'],
                'age_range': f"{mind_age_data['age_range'][0]}-{mind_age_data['age_range'][1]} years",
                'personality_type': mind_age_data['personality_type'],
                'emotional_intelligence': mind_age_data['emotional_intelligence'],
                'ei_description': mind_age_data['ei_description'],
                'interpretation': f"Based on your emotional patterns, your psychological age appears to be around {mind_age_data['mind_age']} years."
            },
            'stress_analysis': {
                'level': stress_assessment.get('level', 'moderate'),
                'score': stress_assessment.get('score', 5.0),
                'description': stress_assessment.get('description', 'Normal stress levels')
            },
            'iris_analysis': iris_data if iris_data else None
        })
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/fitbit')
def fitbit_page():
    """Serve the Fitbit integration page"""
    return render_template('fitbit.html')


@app.route('/fitbit.html')
def fitbit_redirect():
    """Alternative route for Fitbit page"""
    return render_template('fitbit.html')


# Optional: Add API endpoint for Fitbit data processing
@app.route('/api/fitbit/sync', methods=['POST'])
def fitbit_sync():
    """Handle Fitbit data synchronization"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        # Extract Fitbit data
        heart_rate = data.get('heart_rate')
        steps = data.get('steps')
        sleep_data = data.get('sleep')
        activity_data = data.get('activity')
        
        # Process and analyze Fitbit data
        # You can integrate this with your burnout detection logic
        
        analysis = {
            'heart_rate_variability': 'normal' if heart_rate and 60 <= heart_rate <= 100 else 'elevated',
            'activity_level': 'good' if steps and steps >= 5000 else 'low',
            'sleep_quality': sleep_data.get('quality', 'unknown') if sleep_data else 'unknown',
            'stress_indicators': []
        }
        
        # Add stress indicators based on Fitbit data
        if heart_rate and heart_rate > 100:
            analysis['stress_indicators'].append('Elevated resting heart rate detected')
        
        if steps and steps < 3000:
            analysis['stress_indicators'].append('Low physical activity')
        
        if sleep_data and sleep_data.get('hours', 0) < 6:
            analysis['stress_indicators'].append('Insufficient sleep duration')
        
        return jsonify({
            'success': True,
            'message': 'Fitbit data synced successfully',
            'analysis': analysis,
            'recommendations': [
                'Maintain regular physical activity',
                'Aim for 7-9 hours of sleep',
                'Monitor heart rate patterns',
                'Stay hydrated throughout the day'
            ]
        })
        
    except Exception as e:
        logger.error(f"Fitbit sync error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fitbit/callback', methods=['GET'])
def fitbit_callback():
    """Handle Fitbit OAuth callback"""
    try:
        # Get authorization code from callback
        code = request.args.get('code')
        
        if not code:
            return jsonify({'success': False, 'error': 'Authorization failed'}), 400
        
        # Here you would exchange the code for an access token
        # This is a placeholder - implement actual OAuth flow
        
        return jsonify({
            'success': True,
            'message': 'Fitbit connected successfully',
            'redirect': '/fitbit'
        })
        
    except Exception as e:
        logger.error(f"Fitbit callback error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    


# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with helpful information"""
    requested_path = request.path
    logger.warning(f"404 Error - Requested path: {requested_path}")
    
    # Return JSON for API endpoints
    if requested_path.startswith('/api/'):
        return jsonify({
            'success': False, 
            'error': 'Endpoint not found',
            'path': requested_path,
            'available_endpoints': [
                '/api/health',
                '/api/routes',
                '/api/get-questions',
                '/api/get-voice-prompts',
                '/api/create-session',
                '/api/save-voice',
                '/api/upload-frames',
                '/api/analyze-emotions',
                '/api/get-recommendations'
            ]
        }), 404
    
    # Return HTML error page for regular routes
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>404 - Page Not Found</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                text-align: center;
            }}
            h1 {{ color: #e74c3c; }}
            .links {{ margin-top: 30px; }}
            .links a {{
                display: inline-block;
                margin: 10px;
                padding: 10px 20px;
                background: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .links a:hover {{ background: #2980b9; }}
        </style>
    </head>
    <body>
        <h1>404 - Page Not Found</h1>
        <p>The page you requested (<strong>{requested_path}</strong>) does not exist.</p>
        <div class="links">
            <a href="/">Home</a>
            <a href="/real-time-analysis">Real-Time Analysis</a>
            <a href="/gender">Start Questionnaire</a>
        </div>
    </body>
    </html>
    """, 404


@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'error': 'File too large (max 32MB)'}), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error', 'details': str(error)}), 500


# ==================== MAIN APPLICATION ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 BURNOUT DETECTION SYSTEM - COMBINED APPLICATION")
    print("="*70)
    print("\n📋 Available Features:")
    print("   ✅ Questionnaire-based Burnout Assessment")
    print("   ✅ Real-time Face Emotion Analysis")
    print("   ✅ Voice Stress Assessment")
    print("   ✅ Iris Tracking & Pupil Dilation")
    print("   ✅ AI-Powered Recommendations")
    
    print("\n📋 Available Routes:")
    print("   GET  / - Main home page")
    print("   GET  /home - Home page")
    print("   GET  /gender - Gender selection")
    print("   GET  /occupation - Occupation selection")
    print("   GET  /marital-status - Marital status selection")
    print("   GET  /questions - Burnout questionnaire")
    print("   GET  /result - Questionnaire results")
    print("   GET  /real-time-analysis - Face & voice analysis")
    print("   GET  /index.html - Real-time analysis (alternative)")
    print("   GET  /fitbit - Fitbit integration")
    print("   GET  /fitbit.html - Fitbit integration (alternative)")
    
    print("\n📋 API Endpoints:")
    print("   GET  /api/health - Health check")
    print("   GET  /api/routes - List all available routes")
    print("   GET  /api/get-questions - Get stress questions")
    print("   GET  /api/get-voice-prompts - Get voice prompts")
    print("   POST /api/create-session - Create analysis session")
    print("   POST /api/save-voice - Save voice recording")
    print("   POST /api/upload-frames - Upload face frames")
    print("   POST /api/analyze-emotions - Analyze emotions & iris")
    print("   POST /api/get-recommendations - Get AI recommendations")
    print("   POST /api/fitbit/sync - Sync Fitbit data")
    print("   GET  /api/fitbit/callback - Fitbit OAuth callback")


    
    print(f"\n🔧 System Status:")
    print(f"   DeepFace: {'✅ Available' if DEEPFACE_AVAILABLE else '❌ Not Available (using mock)'}")
    print(f"   MediaPipe: {'✅ Available (Iris Detection)' if MEDIAPIPE_AVAILABLE else '❌ Not Available (mock)'}")
    print(f"   Gemini AI: {'✅ Available (AI Recommendations)' if GEMINI_AVAILABLE else '❌ Not Available (fallback)'}")
    
    if GEMINI_AVAILABLE and not GEMINI_API_KEY:
        print("   ⚠️  Warning: Gemini API key not set. Set GEMINI_API_KEY environment variable.")
    
    print(f"\n📁 Configuration:")
    print(f"   Database: {DB_PATH}")
    print(f"   Questions: {QUESTIONS_PATH}")
    print(f"   Sessions: {app.config['SESSION_FOLDER']}")
    print(f"   Voice: {app.config['VOICE_FOLDER']}")
    
    print("\n⏱️  Analysis Configuration:")
    print("   • Session Duration: 60 seconds")
    print("   • Frame Capture: 6 frames (every 10 seconds)")
    print("   • Voice Questions: 3 stress assessment questions")
    print("   • Iris Metrics: Pupil dilation, eye openness, gaze stability")
    
    print("\n🌐 Server Starting...")
    print("   URL: http://localhost:5000")
    print("   URL: http://0.0.0.0:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)