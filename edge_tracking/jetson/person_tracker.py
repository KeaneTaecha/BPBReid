import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
import json
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

class PersonTracker:
    def __init__(self, max_disappeared=15, max_distance=100):
        self.persons = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0

        # ReID configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize YOLO for person detection
        self.initialize_model()
        
        # Initialize ReID components (simplified)
        self.initialize_reid()
        
        # Gallery storage for ReID
        self.gallery_features = None
        self.gallery_loaded = False
        
        # Reidentification parameters
        self.reid_threshold = 0.6  # Adjusted for ResNet features
        self.person_detection_threshold = 0.7
        
        # Frame buffer for new person verification
        self.new_person_buffer = {}
        self.frames_to_check = 4
        self.frames_to_match = 2
        
        self.history_file = "person_history.pkl"
        
        print("Person tracker with ReID initialized successfully")

    def initialize_reid(self):
        """Initialize simplified ReID using ResNet50"""
        try:
            # Use ResNet50 as feature extractor
            self.reid_model = models.resnet50(pretrained=True)
            
            # Remove the final classification layer
            self.reid_model = torch.nn.Sequential(*list(self.reid_model.children())[:-1])
            
            self.reid_model = self.reid_model.to(self.device)
            self.reid_model.eval()
            
            # Setup transforms for ResNet
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.reid_enabled = True
            print("ReID system initialized with ResNet50")
            
        except Exception as e:
            print(f"Error initializing ReID: {e}")
            self.reid_enabled = False

    def load_gallery_person(self, gallery_path):
        """Load gallery person image and extract ReID features"""
        if not self.reid_enabled:
            print("ReID not enabled, skipping gallery loading")
            return False
            
        try:
            print(f"Loading gallery person from: {gallery_path}")
            
            # Load and detect person in gallery image
            image = cv2.imread(gallery_path)
            if image is None:
                print(f"Could not load gallery image: {gallery_path}")
                return False
            
            # Detect person using YOLO
            results = self.model(image, conf=self.person_detection_threshold, classes=[0], verbose=False)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = box.astype(int)
                        person_img = image[y1:y2, x1:x2]
                        
                        if person_img.size > 0:
                            # Extract ReID features
                            self.gallery_features = self.extract_reid_features(person_img)
                            self.gallery_loaded = True
                            print(f"Gallery person loaded successfully. Feature shape: {self.gallery_features.shape}")
                            return True
            
            print("No person detected in gallery image")
            return False
            
        except Exception as e:
            print(f"Error loading gallery person: {e}")
            return False

    def extract_reid_features(self, person_img):
        """Extract ReID features from person image using ResNet50"""
        try:
            # Convert BGR to RGB
            if len(person_img.shape) == 3 and person_img.shape[2] == 3:
                person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            else:
                person_img_rgb = person_img
                
            # Convert to PIL Image
            image = Image.fromarray(person_img_rgb)
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.reid_model(image_tensor)
                features = features.squeeze()
                
                # L2 normalize
                features = F.normalize(features, p=2, dim=0)
                
            return features
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return torch.randn(2048).to(self.device)  # ResNet50 outputs 2048-dim features

    def compute_similarity(self, query_features, gallery_features):
        """Compute cosine similarity between query and gallery features"""
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        if gallery_features.dim() == 1:
            gallery_features = gallery_features.unsqueeze(0)
        
        # Cosine similarity
        similarity = torch.mm(query_features, gallery_features.t())
        
        return similarity.item()

    def check_gallery_match(self, person_img):
        """Check if person matches gallery using ReID"""
        if not self.reid_enabled or not self.gallery_loaded:
            return False, 0.0
        
        try:
            # Extract features from current person
            query_features = self.extract_reid_features(person_img)
            
            # Compute similarity
            similarity = self.compute_similarity(query_features, self.gallery_features)
            
            # Check if match
            is_match = similarity > self.reid_threshold
            
            return is_match, similarity
            
        except Exception as e:
            print(f"Error checking gallery match: {e}")
            return False, 0.0

    def initialize_model(self):
        """Initialize YOLOv11 model using Ultralytics"""
        try:
            engine_path = "yolo11n.engine"

            if os.path.exists(engine_path):
                print(f"Loading existing TensorRT engine from {engine_path}")
                self.model = YOLO(engine_path, task='detect')
            else:
                print("TensorRT engine not found. Creating one")
                self.model = YOLO("yolo11n.pt")
                self.model.export(format="engine", half=True)
                self.model = YOLO("yolo11n.engine", task='detect')

            print("YOLOv11 model initialized successfully")
            self.conf_threshold = 0.7
            self.person_class = 0

        except Exception as e:
            print(f"Error initializing YOLOv11 model: {e}")
            raise

    def detect_persons(self, frame):
        """Detect persons in the frame using YOLOv11"""
        start_time = time.time()
        try:
            results = self.model(frame,
                                conf=self.conf_threshold,
                                classes=[self.person_class],
                                verbose=False)

            boxes = []
            features_list = []

            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                        x1, y1, x2, y2 = box
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        boxes.append([x, y, w, h])
                        features_list.append(None)  # Placeholder

            elapsed_time = time.time() - start_time
            print(f"detect_persons time: {elapsed_time:.4f} seconds")

            return boxes, features_list

        except Exception as e:
            print(f"Error in person detection: {e}")
            return [], []

    def update(self, frame, boxes, extracted_features=None):
        """Update person tracking with new detections and ReID verification"""
        start_time = time.time()

        if len(boxes) == 0:
            print("No Detection")
            for person_id in list(self.persons.keys()):
                self.persons[person_id]["disappeared"] += 1

                if self.persons[person_id]["disappeared"] > self.max_disappeared:
                    history = self.load_history()
                    history[person_id] = {
                        "features": self.persons[person_id].get("features"),
                        "last_seen": time.time(),
                        "bbox": self.persons[person_id]["bbox"]
                    }
                    self.save_history(history)
                    print(f"Person ID {person_id} moved to history")
                    del self.persons[person_id]

            elapsed_time = time.time() - start_time
            print(f"update (No detection) time: {elapsed_time:.4f} seconds")
            return self.persons

        print(f"Processing {len(boxes)} detections")

        centroids = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            centroid = (int(x + w/2), int(y + h/2))
            centroids.append(centroid)

        used_detections = set()

        # Match with existing persons based on spatial proximity
        if len(self.persons) > 0:
            person_ids = list(self.persons.keys())
            
            for person_id in person_ids:
                person = self.persons[person_id]
                prev_centroid = person["centroid"]
                
                # Find closest detection
                min_dist = float('inf')
                closest_idx = -1
                
                for j, centroid in enumerate(centroids):
                    if j not in used_detections:
                        dist = np.sqrt((prev_centroid[0] - centroid[0])**2 + 
                                     (prev_centroid[1] - centroid[1])**2)
                        if dist < min_dist and dist < self.max_distance:
                            min_dist = dist
                            closest_idx = j
                
                if closest_idx >= 0:
                    # Update existing person
                    self.persons[person_id]["centroid"] = centroids[closest_idx]
                    self.persons[person_id]["bbox"] = boxes[closest_idx]
                    self.persons[person_id]["disappeared"] = 0
                    used_detections.add(closest_idx)
                    print(f"Updated person ID {person_id}")
                else:
                    # Mark as disappeared
                    self.persons[person_id]["disappeared"] += 1
                    if self.persons[person_id]["disappeared"] > self.max_disappeared:
                        del self.persons[person_id]
                        print(f"Removed person ID {person_id}")

        # Process new detections with ReID verification
        for j in range(len(boxes)):
            if j in used_detections:
                continue
            
            # Extract person image for ReID
            x, y, w, h = boxes[j]
            person_img = frame[y:y+h, x:x+w]
            
            if person_img.size == 0:
                continue
            
            # Initialize buffer for this detection if needed
            detection_key = f"{x}_{y}_{w}_{h}"
            
            if self.reid_enabled and self.gallery_loaded:
                # Check if we're already tracking this detection
                if detection_key not in self.new_person_buffer:
                    self.new_person_buffer[detection_key] = {
                        'frames_checked': 0,
                        'frames_matched': 0,
                        'centroid': centroids[j],
                        'bbox': boxes[j]
                    }
                
                buffer = self.new_person_buffer[detection_key]
                
                # Check against gallery for first 4 frames
                if buffer['frames_checked'] < self.frames_to_check:
                    is_match, similarity = self.check_gallery_match(person_img)
                    buffer['frames_checked'] += 1
                    
                    if is_match:
                        buffer['frames_matched'] += 1
                        print(f"Frame {buffer['frames_checked']}/{self.frames_to_check}: "
                              f"Match found (similarity: {similarity:.3f})")
                    else:
                        print(f"Frame {buffer['frames_checked']}/{self.frames_to_check}: "
                              f"No match (similarity: {similarity:.3f})")
                    
                    # Check if we've evaluated enough frames
                    if buffer['frames_checked'] >= self.frames_to_check:
                        if buffer['frames_matched'] >= self.frames_to_match:
                            # This is the gallery person - assign special ID 0
                            self.persons[0] = {
                                "centroid": centroids[j],
                                "bbox": boxes[j],
                                "disappeared": 0,
                                "is_gallery": True
                            }
                            print(f"Gallery person detected! Assigned ID 0 "
                                  f"({buffer['frames_matched']}/{self.frames_to_check} frames matched)")
                        else:
                            # Not the gallery person - assign regular ID
                            self.register(centroids[j], boxes[j], None)
                            print(f"New person registered with ID {self.next_id-1} "
                                  f"({buffer['frames_matched']}/{self.frames_to_check} frames matched)")
                        
                        # Clean up buffer
                        del self.new_person_buffer[detection_key]
                
            else:
                # ReID not enabled - use regular registration
                self.register(centroids[j], boxes[j], None)
                print(f"Registered new person with ID {self.next_id-1} (ReID disabled)")

        # Clean up old buffers
        current_detections = {f"{box[0]}_{box[1]}_{box[2]}_{box[3]}" for box in boxes}
        buffer_keys_to_remove = []
        for key in self.new_person_buffer:
            if key not in current_detections:
                buffer_keys_to_remove.append(key)
        for key in buffer_keys_to_remove:
            del self.new_person_buffer[key]

        elapsed_time = time.time() - start_time
        print(f"update time: {elapsed_time:.4f} seconds")
        return self.persons

    def register(self, centroid, bbox, features):
        """Register a new person"""
        existing_ids = set(self.persons.keys())
        history = self.load_history()
        historical_ids = set(history.keys())
        all_ids = existing_ids.union(historical_ids)
        
        if all_ids:
            max_existing_id = max(int(id) for id in all_ids if id != 0)
            if self.next_id <= max_existing_id:
                self.next_id = max_existing_id + 1
        
        # Make sure we don't use ID 0 (reserved for gallery)
        if self.next_id == 0:
            self.next_id = 1
        
        self.persons[self.next_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "disappeared": 0,
            "features": features,
            "is_gallery": False
        }
        print(f"Registered new person with ID {self.next_id}")
        self.next_id += 1

    def save_history(self, history):
        """Save history to file"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(history, f)
            print(f"History saved to {self.history_file}")
        except Exception as e:
            print(f"Error saving history: {e}")

    def load_history(self):
        """Load history from file and return it"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    history = pickle.load(f)
                print(f"History loaded from {self.history_file}, {len(history)} entries")
                return history
            else:
                print("No history file found, starting with empty history")
                return {}
        except Exception as e:
            print(f"Error loading history: {e}")
            return {}