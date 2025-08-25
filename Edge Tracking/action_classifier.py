import cv2
import numpy as np
from ultralytics import YOLO
import time
import math

from datetime import datetime
import pytz
import threading
from twilio_whatsapp import send_whatsapp_alert_with_image
import os

class ActionClassifier:
    def __init__(self):
        # Initialize YOLO11 pose model
        self.initialize_pose_model()

        # Action classification parameters
        self.action_history = {}  # Store recent poses for temporal analysis
        self.history_length = 10  # Number of frames to consider for action classification

        # Define keypoint indices (COCO format)
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

        # Fall detection and WhatsApp alert parameters
        self.last_fall_alert_time = 0
        self.fall_alert_interval = 60

        print("Action Classifier initialized successfully")

    def initialize_pose_model(self):
        """Initialize YOLO11 pose estimation model"""
        try:
            # Check if pose model exists
            pose_model_path = "yolo11n-pose.pt"

            # Load YOLO11 pose model
            self.model = YOLO(pose_model_path)
            print("YOLO11 pose model initialized successfully")

            # Set confidence threshold
            self.conf_threshold = 0.5

        except Exception as e:
            print(f"Error initializing YOLO11 pose model: {e}")
            print("Make sure YOLO11 pose model is installed correctly.")
            raise

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        if any(p is None for p in [p1, p2, p3]):
            return None

        # Calculate vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.degrees(math.acos(cos_angle))

        return angle

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        if p1 is None or p2 is None:
            return None
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_keypoint(self, keypoints, name):
        """Get keypoint coordinates if confidence is high enough"""
        idx = self.keypoint_indices[name]
        if idx * 3 + 2 < len(keypoints):
            x, y, conf = keypoints[idx*3], keypoints[idx*3+1], keypoints[idx*3+2]
            if conf > 0.5:  # Confidence threshold
                return (x, y)
        return None

    def classify_action(self, keypoints, person_id=0, frame=None):
        """Classify action based on pose keypoints - only walking, sitting, standing"""
        if keypoints is None or len(keypoints) < 51:  # 17 keypoints * 3 (x,y,conf)
            return "unknown"

        # Extract key body points
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        left_knee = self.get_keypoint(keypoints, 'left_knee')
        right_knee = self.get_keypoint(keypoints, 'right_knee')
        left_ankle = self.get_keypoint(keypoints, 'left_ankle')
        right_ankle = self.get_keypoint(keypoints, 'right_ankle')

        # Calculate hip center for movement tracking
        if left_hip and right_hip:
            hip_center = ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
        else:
            hip_center = None

        # Store current pose for temporal analysis
        if person_id not in self.action_history:
            self.action_history[person_id] = []

        current_pose = {
            'timestamp': time.time(),
            'keypoints': keypoints,
            'hip_center': hip_center
        }

        self.action_history[person_id].append(current_pose)

        # Keep only recent history
        if len(self.action_history[person_id]) > self.history_length:
            self.action_history[person_id].pop(0)

        # FALLING/LYING DOWN DETECTION (highest priority)
        if self.detect_falling(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee):
            current_time = time.time()

            # Check if enough time has passed since last alert
            if current_time - self.last_fall_alert_time >= self.fall_alert_interval:

                image_path = None
                if frame is not None:
                    folder_name = "fall_detection_images"
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)

                    ny_tz = pytz.timezone('America/New_York')
                    # timestamp = datetime.now(ny_tz).strftime('%Y%m%d_%H%M%S')
                    timestamp = datetime.now(ny_tz).strftime('%Y-%m-%d %H:%M:%S')

                    # Save in the folder
                    image_path = os.path.join(folder_name, f"fall_detection_{timestamp}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Fall image saved: {image_path}")

                send_whatsapp_alert_with_image(image_path, timestamp)
                self.last_fall_alert_time = current_time
            return "falling"

        # SITTING DETECTION
        if self.detect_sitting(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
            return "sitting"

        # WALKING DETECTION
        if len(self.action_history[person_id]) >= 5:
            if self.detect_walking(person_id):
                return "walking"

        # DEFAULT TO STANDING
        return "standing"


    def detect_falling(self, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee):
        """Detect falling/lying down posture with knee angle verification"""
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return False

        # Calculate body orientation - check if torso is horizontal
        shoulder_center = ((left_shoulder[0] + right_shoulder[0])/2,
                          (left_shoulder[1] + right_shoulder[1])/2)
        hip_center = ((left_hip[0] + right_hip[0])/2,
                      (left_hip[1] + right_hip[1])/2)

        # Calculate torso angle relative to vertical
        torso_horizontal_dist = abs(shoulder_center[0] - hip_center[0])
        torso_vertical_dist = abs(shoulder_center[1] - hip_center[1])

        print(f"horizontal: {torso_horizontal_dist}")
        print(f"vertical: {torso_vertical_dist}")

        # If horizontal distance is greater than vertical distance, person might be lying down
        if torso_horizontal_dist > torso_vertical_dist:
            # Verify with knee angles
            if not left_knee or not right_knee:
                # If knee data is missing, rely on torso orientation only
                return False

            # Calculate the body angle (angle of torso from vertical)
            body_angle = math.degrees(math.atan2(torso_horizontal_dist, torso_vertical_dist))

            # Determine falling direction
            falling_right = shoulder_center[0] > hip_center[0]

            # Calculate knee angles relative to the hip
            if falling_right:
                # Person falling to the left
                hip_knee_horizontal = abs(left_hip[0] - left_knee[0])
                hip_knee_vertical = abs(left_hip[1] - left_knee[1])

                knee_angle_from_vertical = math.degrees(math.atan2(hip_knee_horizontal, hip_knee_vertical))

                angle_difference = abs(knee_angle_from_vertical - body_angle)

                if angle_difference > 10:
                    return False

            else:
                # Person falling to the right
                hip_knee_horizontal = abs(right_hip[0] - right_knee[0])
                hip_knee_vertical = abs(right_hip[1] - right_knee[1])

                # Calculate the angle of the knee relative to vertical
                knee_angle_from_vertical = math.degrees(math.atan2(hip_knee_horizontal, hip_knee_vertical))

                angle_difference = abs(knee_angle_from_vertical - body_angle)

                if angle_difference > 10:
                    return False

            return True

        return False


    def detect_sitting(self, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
        """Detect sitting posture"""
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return False

        # Calculate knee angles
        left_knee_angle = None
        right_knee_angle = None

        if left_ankle:
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        if right_ankle:
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate hip-knee angles (angle between hip-knee line and vertical)
        left_hip_knee_angle = None
        right_hip_knee_angle = None

        # Create a vertical reference point below the hip
        left_vertical_ref = (left_hip[0], left_hip[1] + 100)
        right_vertical_ref = (right_hip[0], right_hip[1] + 100)

        left_hip_knee_angle = self.calculate_angle(left_vertical_ref, left_hip, left_knee)
        right_hip_knee_angle = self.calculate_angle(right_vertical_ref, right_hip, right_knee)

        # Sitting: knees are bent and thighs are more horizontal
        sitting_indicators = 0

        # Check: knees should be higher than ankles (relative y-position)
        if left_knee and left_ankle:
            if left_knee_angle and 60 <= left_knee_angle <= 130 and left_knee[1] < left_ankle[1]:
              sitting_indicators += 1
              print(f"Knee Left: {left_knee_angle}")
        elif left_knee_angle and 60 <= left_knee_angle <= 130 or left_hip_knee_angle > 40:
            sitting_indicators += 1
            print(f"Left Hip: {left_hip_knee_angle}")

        if right_knee and right_ankle:
            if right_knee_angle and 60 <= right_knee_angle <= 130 and right_knee[1] < right_ankle[1]:
                sitting_indicators += 1
                print(f"Knee Right: {right_knee_angle}")

        elif right_knee_angle and 60 <= right_knee_angle <= 130 or right_hip_knee_angle > 40:
            sitting_indicators += 1
            print(f"Right Hip: {right_hip_knee_angle}")

        return sitting_indicators >= 1

    def detect_standing(self, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
        """Detect standing posture"""
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return False

        # Calculate leg straightness
        left_leg_straight = False
        right_leg_straight = False

        if left_ankle:
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            if left_knee_angle and left_knee_angle > 130:
                left_leg_straight = True

        if right_ankle:
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            if right_knee_angle and right_knee_angle > 130:
                right_leg_straight = True

        return left_leg_straight or right_leg_straight

    def detect_walking(self, person_id):
        """Detect walking based on temporal movement patterns"""
        if len(self.action_history[person_id]) < 5:
            return False

        recent_poses = self.action_history[person_id][-5:]

        # Calculate movement of hip center over time
        hip_movements = []
        for i in range(1, len(recent_poses)):
            prev_hip = recent_poses[i-1]['hip_center']
            curr_hip = recent_poses[i]['hip_center']

            if prev_hip and curr_hip:
                movement = self.calculate_distance(prev_hip, curr_hip)
                hip_movements.append(movement)

        if not hip_movements:
            return False

        # Walking: consistent movement with some variation
        avg_movement = np.mean(hip_movements)
        movement_std = np.std(hip_movements)

        # Threshold for walking detection
        return avg_movement > 5 and movement_std > 2



    def detect_persons_and_poses(self, frame):
        """Detect persons and their poses using YOLO11"""
        try:
            # Run YOLO11 pose estimation
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

            detections = []

            for result in results:
                # Get bounding boxes and keypoints
                if hasattr(result, 'boxes') and hasattr(result, 'keypoints'):
                    boxes = result.boxes
                    keypoints = result.keypoints

                    if boxes is not None and keypoints is not None:
                        for i in range(len(boxes)):
                            # Get bounding box
                            box = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = box
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                            # Get keypoints
                            kpts = keypoints.xy[i].cpu().numpy().flatten() if len(keypoints.xy) > i else None

                            # Get confidence scores for keypoints
                            if hasattr(keypoints, 'conf') and len(keypoints.conf) > i:
                                kpts_conf = keypoints.conf[i].cpu().numpy().flatten()
                                # Interleave x,y coordinates with confidence scores
                                kpts_with_conf = []
                                for j in range(len(kpts)//2):
                                    kpts_with_conf.extend([kpts[j*2], kpts[j*2+1], kpts_conf[j]])
                                kpts = kpts_with_conf

                            # Classify action based on pose - pass frame for fall alert
                            action = self.classify_action(kpts, person_id=i, frame=frame)

                            detections.append({
                                'bbox': [x, y, w, h],
                                'keypoints': kpts,
                                'action': action,
                                'confidence': float(boxes.conf[i]) if hasattr(boxes, 'conf') else 0.0
                            })

            return detections

        except Exception as e:
            print(f"Error in pose detection: {e}")
            return []

    def draw_pose_and_action(self, frame, detections):
        """Draw pose keypoints and action labels on frame"""
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            keypoints = detection['keypoints']
            action = detection['action']
            confidence = detection['confidence']

            x, y, w, h = bbox

            # Draw bounding box
            color = self.get_action_color(action)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw action label
            label = f"{action} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw keypoints
            if keypoints and len(keypoints) >= 51:  # 17 keypoints * 3
                self.draw_keypoints(frame, keypoints)

        return frame

    def get_action_color(self, action):
        """Get color for each action type"""
        color_map = {
            'walking': (0, 255, 0),      # Green
            'sitting': (255, 0, 0),      # Blue
            'standing': (0, 255, 255),   # Yellow
            'falling': (0, 0, 255),      # Red
            'unknown': (128, 128, 128)   # Gray
        }
        return color_map.get(action, (128, 128, 128))

    def draw_keypoints(self, frame, keypoints):
        """Draw pose keypoints on frame"""
        # Define skeleton connections
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
        ]

        # Draw keypoints
        for i in range(0, len(keypoints), 3):
            if i + 2 < len(keypoints):
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw skeleton
        for connection in skeleton:
            kpt1_idx, kpt2_idx = connection
            if kpt1_idx * 3 + 2 < len(keypoints) and kpt2_idx * 3 + 2 < len(keypoints):
                x1, y1, conf1 = keypoints[kpt1_idx*3], keypoints[kpt1_idx*3+1], keypoints[kpt1_idx*3+2]
                x2, y2, conf2 = keypoints[kpt2_idx*3], keypoints[kpt2_idx*3+1], keypoints[kpt2_idx*3+2]

                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)