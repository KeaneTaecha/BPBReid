import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
import json
import pickle

class PersonTracker:
    def __init__(self, max_disappeared=15, max_distance=100):
        self.persons = {}
        self.max_disappeared = max_disappeared  # Max frames to keep a disappeared person
        self.max_distance = max_distance  # Max pixel distance for identity matching
        self.next_id = 0  # Counter for assigning IDs

        # Feature extractor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Feature matching thresholds
        self.active_matching_threshold = 0.95
        self.reidentification_threshold = 0.065

        # Appearance model parameters
        self.feature_update_rate = 0.1

        self.history_file = "person_history.pkl"

        print("Person tracker initialized with:")
        print(f"- Max disappeared frames: {max_disappeared}")
        print(f"- Max distance for matching: {max_distance}px")
        print(f"- Feature update rate: {self.feature_update_rate}")
        print(f"- Active matching threshold: {self.active_matching_threshold}")
        print(f"- Re-identification threshold: {self.reidentification_threshold}")

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

    def initialize_model(self):
        """Initialize YOLOv11 model using Ultralytics"""
        try:
            # Check if the TensorRT engine already exists
            engine_path = "yolo11n.engine"

            if os.path.exists(engine_path):
                print(f"Loading existing TensorRT engine from {engine_path}")
                self.model = YOLO(engine_path)
            else:
                # If engine doesn't exist, create it
                print("TensorRT engine not found. Creating one")
                # Load YOLOv11n model
                self.model = YOLO("yolo11n.pt")

                # Export to TensorRT
                self.model.export(format="engine", half=True)  # creates 'yolo11n.engine'
                self.model = YOLO("yolo11n.engine")

            print("YOLOv11 model initialized successfully")

            # Set confidence threshold
            self.conf_threshold = 0.7

            # For person class filtering (person is class 0 in COCO)
            self.person_class = 0

        except Exception as e:
            print(f"Error initializing YOLOv11 model: {e}")
            print("Make sure YOLOv11 model is installed correctly.")
            raise

    def detect_persons(self, frame):
        """Detect persons in the frame using YOLOv11"""
        # Start timing
        start_time = time.time()
        try:
            # Run YOLOv11 inference with class filter for persons
            results = self.model(frame,
                                conf=self.conf_threshold,
                                classes=[self.person_class],  # Only detect persons
                                verbose=False)

            # Extract person detections
            boxes = []

            # Initialize features_list here
            features_list = []

            # Process results - get bounding boxes
            for result in results:
                for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                    # Convert from xyxy (x1, y1, x2, y2) to xywh (x, y, width, height) format
                    x1, y1, x2, y2 = box
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    boxes.append([x, y, w, h])

                    try:
                        # Feature extraction method
                        feature_vector = self._extract_features(frame, [x, y, w, h])
                        features_list.append(feature_vector)

                    except Exception as e:
                        print("Extraction Failed")
                        features_list.append(None)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            print(f"detect_persons time: {elapsed_time:.4f} seconds")

            return boxes, features_list

        except Exception as e:
            print(f"Error in person detection: {e}")
            return [], []


    def _extract_features(self, frame, bbox):
        """Extract features from a person's bounding box using color histograms and HOG"""

        # Start timing
        start_time = time.time()

        x, y, w, h = bbox

        # Make sure the bbox is within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        # Crop person and resize for consistent features
        if w <= 0 or h <= 0:
            return None

        person_img = frame[y:y+h, x:x+w]
        if person_img.size == 0:
            return None

        # Resize to standard size
        try:
            # Resize person image
            person_img_resized = cv2.resize(person_img, (64, 128))

            # Split into upper, middle and lower body for better clothing recognition
            upper_body = person_img_resized[0:40, :]
            middle_body = person_img_resized[40:80, :]
            lower_body = person_img_resized[80:128, :]

            bins = 12
            ranges = [0, 256]
            channels = [0, 1, 2]  # RGB channels

            # Calculate histograms for each body part
            hist_upper = cv2.calcHist([upper_body], channels, None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            hist_middle = cv2.calcHist([middle_body], channels, None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            hist_lower = cv2.calcHist([lower_body], channels, None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

            # Normalize histograms
            cv2.normalize(hist_upper, hist_upper, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_middle, hist_middle, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_lower, hist_lower, 0, 1, cv2.NORM_MINMAX)

            color_features = np.concatenate([
                hist_upper.flatten() * 0.3,
                hist_middle.flatten() * 0.5,
                hist_lower.flatten() * 0.2
            ])


            # Method 2: HOG features (shape-based)
            win_size = (64, 128)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9

            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            hog_features = hog.compute(person_img_resized).flatten()

            hog_features_reduced = []
            step = 6  # Take every 6th feature
            for i in range(0, len(hog_features), step):
                block_avg = np.mean(hog_features[i:i+step])
                hog_features_reduced.append(block_avg)

            hog_features = np.array(hog_features_reduced)

            feature_vector = np.concatenate([color_features * 0.45, hog_features * 0.55])

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            print(f"Feature extraction time: {elapsed_time:.4f} seconds")

            return feature_vector
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None


    def _calculate_feature_distances_vectorized(self, person_features, features_list, debug_output=False):
        """Vectorized calculation of distances between one person and multiple detections"""

        # Start timing
        start_time = time.time()

        # Filter out None values
        valid_features = [f for f in features_list if f is not None]

        if not valid_features:
            return [], []

        # Convert to numpy arrays if they aren't already
        person_features = np.array(person_features)
        features_array = np.array(valid_features)

        # Extract color histogram and HOG portions
        color_length = 12**3 * 3

        # Extract components for all features at once
        hist_person = person_features[:color_length]
        hist_detections = features_array[:, :color_length]

        hog_person = person_features[color_length:]
        hog_detections = features_array[:, color_length:]

        # Vectorized chi-square calculation
        epsilon = 1e-5
        numerator = (hist_person.reshape(1, -1) - hist_detections)**2
        denominator = hist_person.reshape(1, -1) + hist_detections + epsilon
        chi_square_dists = np.sum(numerator / denominator, axis=1)
        chi_square_dists = np.minimum(1.0, chi_square_dists / 15.0)

        # Vectorized Euclidean distance for HOG
        hog_dists = np.linalg.norm(hog_person.reshape(1, -1) - hog_detections, axis=1)
        hog_dists = hog_dists / (np.sqrt(len(hog_person)) + epsilon)

        # Adaptive weighting (simplified for vectorization)
        hist_vars = np.var(hist_detections, axis=1) + np.var(hist_person)
        color_weights = np.where(hist_vars < 0.05, 0.3, 0.6)
        hog_weights = 1.0 - color_weights

        # Calculate combined distances
        combined_distances = color_weights * chi_square_dists + hog_weights * hog_dists

        # Create a mapping of indices to distances
        valid_indices = [i for i, f in enumerate(features_list) if f is not None]

        # Debug output
        if debug_output:
            print(f"Chi-square dist: {chi_square_dists:.4f}, HOG dist: {hog_dists:.4f}, Combined: {combined_distances:.4f}")

        return valid_indices, combined_distances


    def update(self, frame, boxes, extracted_features=None):
        """Update person tracking with new detections"""

        # Start timing
        start_time = time.time()

        # If no boxes, mark all as disappeared
        if len(boxes) == 0:
            print("No Detection")
            for person_id in list(self.persons.keys()):
                self.persons[person_id]["disappeared"] += 1

                if self.persons[person_id]["disappeared"] > self.max_disappeared:
                    # Load current history from file
                    history = self.load_history()

                    # Add to history
                    history[person_id] = {
                        "features": self.persons[person_id]["features"],
                        "last_seen": time.time(),
                        "bbox": self.persons[person_id]["bbox"]
                    }

                    # Save back to file
                    self.save_history(history)
                    print(f"Person ID {person_id} moved to history")
                    del self.persons[person_id]


            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            print(f"update (No detection) time: {elapsed_time:.4f} seconds")
            return self.persons

        # Debug information
        print(f"Processing {len(boxes)} detections")

        # Extract centroids and features of current detections
        centroids = []
        features_list = []

        for i, box in enumerate(boxes):
            x, y, w, h = box
            centroid = (int(x + w/2), int(y + h/2))
            centroids.append(centroid)

            # Use pre-extracted features if provided, otherwise extract them now
            if extracted_features is not None and i < len(extracted_features):
                features = extracted_features[i]
                features_list.append(features)

        # Track which detections have been matched
        used_detections = set()

        # PART 1: Match with active persons first
        if len(self.persons) > 0:
            # Get IDs and centroids of existing persons
            person_ids = list(self.persons.keys())
            used_persons = set()

            # Create cost matrix for matching
            cost_matrix = np.zeros((len(person_ids), len(boxes)))

            # Fill cost matrix with feature distances - vectorized approach
            for i, person_id in enumerate(person_ids):
                person = self.persons[person_id]

                # Set all distances to infinity initially
                cost_matrix[i, :] = float('inf')

                # Vectorized calculation for all features at once
                valid_indices, distances = self._calculate_feature_distances_vectorized(person["features"], features_list)

                # Apply threshold and fill cost matrix
                for idx, j in enumerate(valid_indices):
                    if distances[idx] <= self.active_matching_threshold:
                        cost_matrix[i, j] = distances[idx]


            # Sort all person-detection pairs by cost
            pairs = []
            for i in range(len(person_ids)):
                for j in range(len(boxes)):
                    if cost_matrix[i, j] != float('inf'):
                        pairs.append((i, j, cost_matrix[i, j]))

            # Sort by cost (lowest first)
            pairs.sort(key=lambda x: x[2])

            # Match in order of lowest cost
            for i, j, cost in pairs:
                if i in used_persons or j in used_detections:
                    continue

                person_id = person_ids[i]

                # It's a match - update the person
                self.persons[person_id]["centroid"] = centroids[j]
                self.persons[person_id]["bbox"] = boxes[j]
                self.persons[person_id]["disappeared"] = 0

                # Update features with moving average for continual adaptation
                if features_list[j] is not None:
                    self.persons[person_id]["features"] = (1 - self.feature_update_rate) * self.persons[person_id]["features"] + self.feature_update_rate * features_list[j]

                used_persons.add(i)
                used_detections.add(j)
                print(f"Matched person ID {person_id} with detection {j}, cost={cost:.4f}")

            # Update disappeared counts for unmatched persons
            for i, person_id in enumerate(person_ids):
                if i not in used_persons:
                    self.persons[person_id]["disappeared"] += 1
                    print(f"Person ID {person_id} marked as disappeared ({self.persons[person_id]['disappeared']}/{self.max_disappeared})")

                    if self.persons[person_id]["disappeared"] > self.max_disappeared:
                        # Load current history from file
                        history = self.load_history()

                        # Add to history
                        history[person_id] = {
                            "features": self.persons[person_id]["features"],
                            "last_seen": time.time(),
                            "bbox": self.persons[person_id]["bbox"]
                        }

                        # Save back to file
                        self.save_history(history)
                        print(f"Person ID {person_id} moved to history")
                        del self.persons[person_id]

        # PART 2: Re-identify from history for remaining detections
        # Process each unmatched detection
        for j in range(len(boxes)):
            if j in used_detections or features_list[j] is None:
                continue

            history = self.load_history()

            # Force re-identification check - crucial debug print to verify this section executes
            print(f"Starting reidentification for detection {j}...")
            print(f"History entries available: {len(history)}")

            # Find the best match in history
            best_match = None
            best_dist = float('inf')

            # Print the detection we're trying to match
            print(f"Detection {j} box: {boxes[j]}")

            # Check against history
            for hist_id, hist_data in history.items():

                # Filter based on size similarity
                hist_box = hist_data["bbox"]
                curr_box = boxes[j]

                # Print comparison for debugging
                print(f"Comparing with history ID {hist_id}, box: {hist_box}")

                # Compare box aspect ratios
                hist_ratio = hist_box[2] / max(hist_box[3], 1)  # width/height
                curr_ratio = curr_box[2] / max(curr_box[3], 1)

                # Skip if aspect ratios are too different
                if abs(hist_ratio - curr_ratio) > 0.7:
                    print(f"Skipping history ID {hist_id} - aspect ratio mismatch: {hist_ratio:.2f} vs {curr_ratio:.2f}")
                    continue

                # Compare areas
                hist_area = hist_box[2] * hist_box[3]
                curr_area = curr_box[2] * curr_box[3]
                area_ratio = max(hist_area, curr_area) / max(min(hist_area, curr_area), 1)

                # Skip if size difference is too large
                if area_ratio > 7:
                    print(f"Skipping history ID {hist_id} - size mismatch: ratio {area_ratio:.2f}")
                    continue


                # Calculate feature distance for appearance matching
                # We need to wrap the single feature in a list to make it work with the vectorized function
                single_feature_list = [features_list[j]]  # Put the single feature in a list
                result = self._calculate_feature_distances_vectorized(hist_data["features"], single_feature_list)

                # Check the type of result and extract correctly
                if isinstance(result, tuple):
                    # If it's returning a tuple (probably (valid_indices, distances))
                    # We want just the distance value for the first item
                    valid_indices, combined_distances = result
                    if len(combined_distances) > 0:
                        feature_dist = float(combined_distances[0])
                    else:
                        feature_dist = float('inf')
                elif isinstance(result, (list, np.ndarray)):
                    # If it's an array/list, take the first element
                    feature_dist = float(result[0])
                else:
                    # If it's already a scalar
                    feature_dist = float(result)

                print(f"History ID {hist_id} feature distance: {feature_dist:.4f}, threshold: {self.reidentification_threshold:.4f}")

                # Threshold for reappearances
                if feature_dist < best_dist and feature_dist < self.reidentification_threshold:
                    best_dist = feature_dist
                    best_match = hist_id
                    print(f"New best match: history ID {hist_id} with distance {feature_dist:.4f}")

            # Apply the best match if found
            if best_match is not None:
                # Re-register with original ID
                self.persons[best_match] = {
                    "centroid": centroids[j],
                    "bbox": boxes[j],
                    "disappeared": 0,
                    "features": features_list[j]
                }
                print(f"Re-identified person ID {best_match} from history, distance={best_dist:.4f}")

                # Remove from history file
                if best_match in history:
                    del history[best_match]
                    self.save_history(history)


                used_detections.add(j)
            else:
                # Register as new person
                self.register(centroids[j], boxes[j], features_list[j])
                print(f"Registered new person with ID {self.next_id-1}")
                used_detections.add(j)

        # Log current tracking status
        history = self.load_history()
        print(f"Currently tracking {len(self.persons)} persons, {len(history)} in history")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"update time: {elapsed_time:.4f} seconds")
        return self.persons

    def register(self, centroid, bbox, features):
      """Register a new person"""
      # Ensure next_id is always higher than any existing ID (including those in history)
      existing_ids = set(self.persons.keys())

      # Load history to check historical IDs as well
      history = self.load_history()
      historical_ids = set(history.keys())

      # Combine all known IDs
      all_ids = existing_ids.union(historical_ids)

      # If there are existing IDs, make sure next_id is higher than all of them
      if all_ids:
          # Convert IDs to integers (in case they're stored as strings)
          max_existing_id = max(int(id) for id in all_ids)
          if self.next_id <= max_existing_id:
              self.next_id = max_existing_id + 1

      # Now register with the guaranteed unique ID
      self.persons[self.next_id] = {
          "centroid": centroid,
          "bbox": bbox,
          "disappeared": 0,
          "features": features
      }
      print(f"Registered new person with ID {self.next_id}")
      self.next_id += 1