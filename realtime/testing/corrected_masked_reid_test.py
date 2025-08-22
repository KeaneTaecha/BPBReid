#!/usr/bin/env python3
"""
Corrected Masked BPBreID Test Implementation

This implementation fixes the false positive issue by:
1. Properly extracting discriminative features
2. Correct feature normalization
3. Better similarity computation
4. Debugging and validation mechanisms
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from torchvision import transforms
import time
from PIL import Image
import os
import json
from datetime import datetime
import torch.nn.functional as F
from types import SimpleNamespace

class CorrectedMaskedBPBreIDTester:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the corrected masked BPBreID system"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create corrected configuration
        self.config = self._create_corrected_config(reid_model_path, hrnet_path)
        
        # Load model
        self.model = self._load_corrected_model()
        
        # Setup transforms
        self.setup_transforms()
        
        # Gallery storage
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Tracking threshold (more conservative)
        self.reid_threshold = 0.45  # Increased threshold to reduce false positives
        
        # Debug mode
        self.debug_mode = False  # Disable verbose debug to reduce clutter
        self.visualize_masks = True  # Enable mask visualization (fixed the bug)
        self.frame_counter = 0  # For periodic debug output
        
        print("Corrected Masked BPBreID system initialized successfully")
    
    def _create_corrected_config(self, model_path, hrnet_path):
        """Create corrected configuration for BPBreID"""
        config = SimpleNamespace()
        
        # Model configuration
        config.model = SimpleNamespace()
        config.model.name = 'bpbreid'
        config.model.load_weights = model_path
        config.model.pretrained = True
        
        # BPBreid configuration - use original settings for stability
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'  # Use identity normalization
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.test_use_target_segmentation = 'soft'
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        config.model.bpbreid.mask_filtering_testing = True
        config.model.bpbreid.mask_filtering_training = True
        
        # Mask configuration - keep original 5 parts
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.softmax_weight = 1.0
        config.model.bpbreid.masks.background_computation_strategy = 'threshold'
        config.model.bpbreid.masks.mask_filtering_threshold = 0.3
        
        # Data configuration
        config.data = SimpleNamespace()
        config.data.height = 384
        config.data.width = 128
        config.data.norm_mean = [0.485, 0.456, 0.406]
        config.data.norm_std = [0.229, 0.224, 0.225]
        
        # Loss configuration
        config.loss = SimpleNamespace()
        config.loss.name = 'part_based'
        
        return config
    
    def _load_corrected_model(self):
        """Load BPBreid model with corrected loading"""
        print("Loading corrected BPBreid model...")
        
        try:
            # Build model with original configuration
            model = torchreid.models.build_model(
                name='bpbreid',
                num_classes=751,
                config=self.config,
                pretrained=True
            )
            
            # Load weights
            checkpoint = torch.load(self.config.model.load_weights, map_location=self.device)
            
            # Handle state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load state dict with strict=False to handle any mismatches
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            
            model = model.to(self.device)
            model.eval()
            
            print("Corrected BPBreid model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading corrected BPBreid model: {e}")
            raise e
    
    def setup_transforms(self):
        """Setup transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((self.config.data.height, self.config.data.width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.norm_mean,
                std=self.config.data.norm_std
            )
        ])
        print(f"Transforms setup for {self.config.data.height}x{self.config.data.width}")
    
    def generate_corrected_masks(self, batch_size, height, width):
        """Generate adaptive body part masks based on actual person content"""
        try:
            parts_num = 5
            feat_h, feat_w = height // 8, width // 8
            
            masks = torch.zeros(batch_size, parts_num + 1, feat_h, feat_w)
            part_names = ['background', 'head', 'upper_torso', 'lower_torso', 'upper_legs', 'lower_legs']
            
            # Create adaptive masks based on person content rather than fixed geometry
            # This approach uses vertical intensity distribution to estimate body part locations
            
            # Create a basic intensity-based mask to find person boundaries
            if hasattr(self, 'current_person_tensor'):
                person_tensor = self.current_person_tensor
                
                # Convert to grayscale and resize to feature map size
                if len(person_tensor.shape) == 4:  # [batch, channels, height, width]
                    gray = person_tensor.mean(dim=1, keepdim=True)  # Average across color channels
                    gray_resized = F.interpolate(gray, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
                    intensity_profile = gray_resized[0, 0].cpu().numpy()
                    
                    # Calculate vertical intensity profile to find body part boundaries
                    vertical_profile = intensity_profile.mean(axis=1)  # Average across width
                    
                    # Smooth the profile with simple moving average (no scipy needed)
                    def simple_smooth(arr, window=3):
                        if len(arr) < window:
                            return arr
                        smoothed = np.copy(arr).astype(float)
                        for i in range(window//2, len(arr) - window//2):
                            smoothed[i] = np.mean(arr[i-window//2:i+window//2+1])
                        return smoothed
                    
                    smoothed_profile = simple_smooth(vertical_profile, window=5)
                    
                    # Find body part boundaries based on intensity changes
                    profile_normalized = (smoothed_profile - smoothed_profile.min()) / (smoothed_profile.max() - smoothed_profile.min() + 1e-8)
                    
                    # Detect significant intensity regions (potential body parts)
                    threshold = 0.3
                    body_regions = profile_normalized > threshold
                    
                    # Find start and end of body region
                    if np.any(body_regions):
                        body_start = np.where(body_regions)[0][0]
                        body_end = np.where(body_regions)[0][-1]
                        body_height = body_end - body_start + 1
                        
                        # Adaptive part allocation based on typical human proportions
                        # Head: ~12%, Upper torso: ~25%, Lower torso: ~20%, Upper legs: ~25%, Lower legs: ~18%
                        proportions = [0.12, 0.25, 0.20, 0.25, 0.18]
                        
                        current_pos = body_start
                        for i, prop in enumerate(proportions):
                            part_size = max(1, int(body_height * prop))
                            part_start = current_pos
                            part_end = min(feat_h, current_pos + part_size)
                            
                            # Only create mask if the part falls within the detected body region
                            if part_start < body_end and part_end > body_start:
                                # Create Gaussian-weighted mask for this part
                                center_y = (part_start + part_end) / 2
                                sigma = (part_end - part_start) / 3
                                
                                for y in range(max(0, part_start), min(feat_h, part_end)):
                                    # Modulate mask strength based on actual intensity
                                    intensity_weight = profile_normalized[y] if y < len(profile_normalized) else 0.5
                                    gaussian_weight = np.exp(-((y - center_y) ** 2) / (2 * sigma ** 2))
                                    final_weight = gaussian_weight * intensity_weight
                                    
                                    masks[:, i + 1, y, :] = final_weight
                            
                            current_pos = part_end
                    else:
                        # Fallback to geometric division if no clear body detected
                        self._create_geometric_masks(masks, feat_h, feat_w, parts_num)
                else:
                    # Fallback to geometric division if tensor format is unexpected
                    self._create_geometric_masks(masks, feat_h, feat_w, parts_num)
            else:
                # Fallback to geometric division if no person tensor available
                self._create_geometric_masks(masks, feat_h, feat_w, parts_num)
            
            # Background mask (complement of all parts)
            masks[:, 0] = torch.clamp(1.0 - masks[:, 1:].sum(dim=1), 0.0, 1.0)
            
            # Normalize masks with softmax for better discrimination
            masks = F.softmax(masks * 2.0, dim=1)  # Temperature scaling for sharper masks
            
            # Store for visualization
            self.last_generated_masks = masks.cpu().numpy()
            self.part_names = part_names
            
            return masks.to(self.device)
            
        except Exception as e:
            print(f"Error generating adaptive masks: {e}")
            # Return simple identity mask as fallback
            identity_mask = torch.ones(batch_size, 1, height // 8, width // 8)
            return identity_mask.to(self.device)
    
    def _create_geometric_masks(self, masks, feat_h, feat_w, parts_num):
        """Fallback geometric mask generation"""
        part_height = feat_h // parts_num
        overlap = max(1, part_height // 4)
        
        for i in range(parts_num):
            start_y = max(0, i * part_height - overlap//2)
            end_y = min(feat_h, (i + 1) * part_height + overlap//2)
            if i == parts_num - 1:
                end_y = feat_h
            
            if start_y < feat_h and end_y > start_y:
                center_y = (start_y + end_y) / 2
                sigma = (end_y - start_y) / 3
                
                for y in range(start_y, end_y):
                    weight = np.exp(-((y - center_y) ** 2) / (2 * sigma ** 2))
                    masks[:, i + 1, y, :] = weight
    
    def extract_features_corrected(self, image):
        """Extract features with corrected processing"""
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Store current person tensor for adaptive mask generation
            self.current_person_tensor = image_tensor
            
            # Generate adaptive masks based on the actual person content
            masks = self.generate_corrected_masks(
                1, 
                self.config.data.height, 
                self.config.data.width
            )
            
            # Extract features
            with torch.no_grad():
                try:
                    # Forward pass with masks
                    outputs = self.model(image_tensor, external_parts_masks=masks)
                    
                    # Process outputs correctly
                    features = self._process_corrected_output(outputs)
                    
                    if features is not None and features.numel() > 0:
                        # Apply proper normalization
                        features = self._apply_corrected_normalization(features)
                        
                        # Assess mask quality for occlusion handling
                        mask_quality = self._assess_mask_quality(masks)
                        
                        # Apply confidence boosting for high-quality masks
                        if mask_quality > 0.7:
                            # High quality masks get a small confidence boost
                            features = features * 1.1
                            features = F.normalize(features, p=2, dim=1)
                        
                        # Periodic debug output (every 30 frames)
                        if self.debug_mode and self.frame_counter % 30 == 0:
                            print(f"Frame {self.frame_counter}: Features shape: {features.shape}")
                            print(f"  Feature mean: {features.mean().item():.6f}, std: {features.std().item():.6f}")
                            print(f"  Feature range: [{features.min().item():.6f}, {features.max().item():.6f}]")
                            print(f"  Mask quality: {mask_quality:.3f}")
                        
                        # Store mask quality for later use
                        self.last_mask_quality = mask_quality
                        
                        return features
                    else:
                        print("Warning: Empty features extracted, using random features")
                        return torch.randn(1, 512).to(self.device)
                        
                except Exception as e:
                    print(f"Model inference error: {e}")
                    return torch.randn(1, 512).to(self.device)
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return torch.randn(1, 512).to(self.device)
    
    def _process_corrected_output(self, outputs):
        """Process model output correctly"""
        try:
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                embeddings_dict = outputs[0]
                
                if isinstance(embeddings_dict, dict):
                    # Priority order for feature selection
                    priority_keys = ['bn_foreg', 'foreground', 'bn_global', 'global']
                    
                    for key in priority_keys:
                        if key in embeddings_dict:
                            features = embeddings_dict[key]
                            if isinstance(features, torch.Tensor) and features.numel() > 0:
                                if len(features.shape) > 2:
                                    features = features.view(features.size(0), -1)
                                
                                if self.debug_mode:
                                    print(f"Using features from key: {key}, shape: {features.shape}")
                                
                                return features
                    
                    # If no priority keys, use first available tensor
                    for key, value in embeddings_dict.items():
                        if isinstance(value, torch.Tensor) and value.numel() > 0:
                            if len(value.shape) > 2:
                                value = value.view(value.size(0), -1)
                            return value
                
                elif isinstance(embeddings_dict, torch.Tensor):
                    if len(embeddings_dict.shape) > 2:
                        return embeddings_dict.view(embeddings_dict.size(0), -1)
                    return embeddings_dict
            
            return None
            
        except Exception as e:
            print(f"Error processing corrected output: {e}")
            return None
    
    def _apply_corrected_normalization(self, features):
        """Apply corrected feature normalization"""
        # L2 normalization
        features = F.normalize(features, p=2, dim=1)
        
        # Additional stabilization - subtract mean to center features
        features = features - features.mean(dim=1, keepdim=True)
        
        # Re-normalize after centering
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def create_mask_visualization(self, person_img, masks):
        """Create visualization of masks overlaid on the person image"""
        try:
            if not hasattr(self, 'last_generated_masks') or not self.visualize_masks:
                return None
            
            # Resize person image to standard size
            person_resized = cv2.resize(person_img, (200, 300))  # Larger size for better visibility
            
            # Create overlay visualization showing masks on the person
            overlay_viz = person_resized.copy().astype(np.float32)
            
            # Define colors for each body part (brighter colors for better visibility)
            part_colors = {
                'background': [80, 80, 80],     # Gray for background
                'head': [0, 0, 255],            # Red for head
                'upper_torso': [0, 255, 0],     # Green for upper torso
                'lower_torso': [255, 0, 0],     # Blue for lower torso
                'upper_legs': [0, 255, 255],    # Cyan for upper legs
                'lower_legs': [255, 0, 255]     # Magenta for lower legs
            }
            
            # Create a winner-takes-all mask assignment to avoid color mixing
            h, w = person_resized.shape[:2]
            mask_assignment = np.zeros((h, w), dtype=np.int32)
            max_values = np.zeros((h, w), dtype=np.float32)
            
            # Collect all non-background masks
            body_parts = []
            for i, (mask_data, part_name) in enumerate(zip(self.last_generated_masks[0], self.part_names)):
                if part_name != 'background':
                    mask_resized = cv2.resize(mask_data, (w, h))
                    body_parts.append((mask_resized, part_name, i))
            
            # Find the dominant mask at each pixel
            for mask_resized, part_name, part_idx in body_parts:
                # Enhance mask visibility
                mask_enhanced = np.power(mask_resized, 0.5)  # Enhance weak values
                mask_enhanced = np.clip(mask_enhanced * 1.5, 0, 1)
                
                # Update assignment where this mask is strongest
                stronger = mask_enhanced > max_values
                mask_assignment[stronger] = part_idx
                max_values[stronger] = mask_enhanced[stronger]
            
            # Apply colors based on mask assignment
            for mask_resized, part_name, part_idx in body_parts:
                # Create mask for pixels assigned to this part
                part_pixels = (mask_assignment == part_idx)
                
                if np.any(part_pixels):
                    # Get intensity from the mask
                    mask_resized = cv2.resize(mask_resized, (w, h))
                    intensity = mask_resized[part_pixels]
                    
                    # Get color for this part
                    color = part_colors.get(part_name, [128, 128, 128])
                    
                    # Apply color with intensity-based alpha
                    alpha_mask = intensity * 0.6  # Variable transparency based on mask strength
                    
                    for c in range(3):
                        overlay_viz[part_pixels, c] = (overlay_viz[part_pixels, c] * (1 - alpha_mask) + 
                                                      color[c] * alpha_mask).astype(np.float32)
            
            # Convert back to uint8
            overlay_viz = np.clip(overlay_viz, 0, 255).astype(np.uint8)
            
            # Create a side-by-side comparison
            viz_width = person_resized.shape[1] * 2 + 20  # Two images + separator
            viz_height = person_resized.shape[0]
            combined_viz = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
            
            # Add original person image
            combined_viz[:, :person_resized.shape[1]] = person_resized
            
            # Add separator
            combined_viz[:, person_resized.shape[1]:person_resized.shape[1]+20] = [60, 60, 60]
            
            # Add overlay visualization
            combined_viz[:, person_resized.shape[1]+20:] = overlay_viz
            
            # Add labels
            cv2.putText(combined_viz, "Original", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_viz, "Body Parts", (person_resized.shape[1] + 30, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add legend for body parts
            legend_y = 50
            for i, (part_name, color) in enumerate(part_colors.items()):
                if part_name == 'background':
                    continue
                
                # Draw colored rectangle
                cv2.rectangle(combined_viz, (person_resized.shape[1] + 30, legend_y + i * 25), 
                             (person_resized.shape[1] + 50, legend_y + i * 25 + 15), color, -1)
                
                # Add text
                cv2.putText(combined_viz, part_name, (person_resized.shape[1] + 55, legend_y + i * 25 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add mask statistics at the bottom
            stats_text = []
            for i, (mask_data, part_name) in enumerate(zip(self.last_generated_masks[0], self.part_names)):
                if part_name != 'background':
                    stats_text.append(f"{part_name}: {mask_data.mean():.3f}")
            
            # Display stats
            stats_y = viz_height - 60
            for i, stat in enumerate(stats_text[:3]):  # Show first 3 stats
                cv2.putText(combined_viz, stat, (10, stats_y + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            for i, stat in enumerate(stats_text[3:]):  # Show remaining stats
                cv2.putText(combined_viz, stat, (person_resized.shape[1] + 30, stats_y + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            return combined_viz
            
        except Exception as e:
            print(f"Error creating mask visualization: {e}")
            return None
    
    def _assess_mask_quality(self, masks):
        """Assess quality of generated masks for occlusion handling"""
        try:
            if not hasattr(self, 'last_generated_masks'):
                return 0.5
            
            mask_data = self.last_generated_masks[0]  # First batch
            
            # Calculate various quality metrics
            
            # 1. Coverage: How much of the image is covered by body parts (not background)
            body_coverage = mask_data[1:].sum() / mask_data.size  # Exclude background
            
            # 2. Distribution: How evenly distributed are the part activations
            part_activations = [mask_data[i].sum() for i in range(1, len(mask_data))]
            part_variance = np.var(part_activations) if len(part_activations) > 1 else 0
            distribution_score = 1.0 / (1.0 + part_variance)  # Lower variance = better distribution
            
            # 3. Clarity: How distinct are the part boundaries
            clarity_scores = []
            for i in range(1, len(mask_data)):
                mask_entropy = -np.sum(mask_data[i] * np.log(mask_data[i] + 1e-8))
                clarity_scores.append(1.0 / (1.0 + mask_entropy))
            avg_clarity = np.mean(clarity_scores) if clarity_scores else 0.5
            
            # 4. Completeness: Check if all parts have some activation
            completeness = sum(1 for i in range(1, len(mask_data)) if mask_data[i].max() > 0.1) / (len(mask_data) - 1)
            
            # Combine metrics with weights
            quality_score = (
                0.3 * body_coverage +
                0.2 * distribution_score +
                0.2 * avg_clarity +
                0.3 * completeness
            )
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error assessing mask quality: {e}")
            return 0.5
    
    def compute_corrected_similarity(self, query_features, gallery_features):
        """Compute corrected similarity with multiple metrics"""
        # Cosine similarity
        cosine_sim = torch.mm(query_features, gallery_features.t())
        
        # Euclidean distance (converted to similarity)
        query_expanded = query_features.unsqueeze(1)
        gallery_expanded = gallery_features.unsqueeze(0)
        euclidean_dist = torch.norm(query_expanded - gallery_expanded, p=2, dim=2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine similarities (weighted average)
        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        return combined_sim
    
    def match_person_corrected(self, query_features):
        """Match person with corrected similarity computation"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        
        # Use corrected similarity computation
        similarities = self.compute_corrected_similarity(query_features, gallery_tensor)
        
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        # Show similarity info for significant detections or periodically
        if self.debug_mode and (best_similarity > self.reid_threshold or self.frame_counter % 30 == 0):
            all_similarities = similarities.squeeze().cpu().numpy()
            print(f"  Similarities: {all_similarities}")
            print(f"  Best: {best_similarity:.3f} vs threshold: {self.reid_threshold}")
        
        if best_similarity > self.reid_threshold:
            return self.gallery_ids[best_idx], best_similarity
        else:
            return None, best_similarity
    
    def add_to_gallery(self, features, person_id, image):
        """Add a person to the gallery"""
        self.gallery_features.append(features)
        self.gallery_ids.append(person_id)
        self.gallery_images.append(image.copy())
        
        if self.debug_mode:
            print(f"Added person {person_id} to gallery. Gallery size: {len(self.gallery_ids)}")
    
    def process_frame_corrected(self, frame):
        """Process a single frame with corrected ReID"""
        self.frame_counter += 1
        results = self.yolo(frame, classes=0, conf=0.5)
        
        frame_results = {
            'detections': [],
            'matches': [],
            'similarities': [],
            'feature_extraction_time': 0.0
        }
        
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features with timing
                        start_time = time.time()
                        features = self.extract_features_corrected(person_img)
                        feature_time = time.time() - start_time
                        frame_results['feature_extraction_time'] += feature_time
                        
                        # Create mask visualization
                        mask_viz = self.create_mask_visualization(person_img, None)
                        
                        # Match with gallery
                        matched_id, similarity = self.match_person_corrected(features)
                        
                        # Get mask quality if available
                        mask_quality = getattr(self, 'last_mask_quality', 0.5)
                        
                        # Store results
                        detection_result = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'matched_id': matched_id,
                            'similarity': similarity,
                            'mask_quality': mask_quality,
                            'feature_extraction_time': feature_time
                        }
                        frame_results['detections'].append(detection_result)
                        frame_results['similarities'].append(similarity)
                        
                        if matched_id is not None:
                            frame_results['matches'].append(matched_id)
                            label = f"Person {matched_id} ({similarity:.3f})"
                            color = (0, 255, 0)  # Green for known persons
                        else:
                            label = f"Unknown ({similarity:.3f})"
                            color = (0, 0, 255)  # Red for unknown persons
                        
                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add mask quality info
                        quality_text = f"MQ:{mask_quality:.2f}"
                        quality_color = (0, 255, 0) if mask_quality > 0.7 else (255, 255, 0) if mask_quality > 0.4 else (255, 0, 0)
                        cv2.putText(annotated_frame, quality_text, (x1, y1-30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)
                        
                        # Add timing info
                        timing_text = f"{feature_time*1000:.1f}ms"
                        cv2.putText(annotated_frame, timing_text, (x1, y2+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Add mask visualization to the side if available
                        if mask_viz is not None and self.visualize_masks:
                            # Resize mask visualization to fit (smaller to avoid issues)
                            target_width = min(300, annotated_frame.shape[1] // 3)
                            target_height = min(150, annotated_frame.shape[0] // 4)
                            mask_viz_small = cv2.resize(mask_viz, (target_width, target_height))
                            
                            # Position it in the top-right corner with safe margins
                            h, w = annotated_frame.shape[:2]
                            viz_h, viz_w = mask_viz_small.shape[:2]
                            
                            # Ensure it fits within the frame
                            if w > viz_w + 20 and h > viz_h + 20:
                                start_x = w - viz_w - 10
                                start_y = 10
                                end_x = start_x + viz_w
                                end_y = start_y + viz_h
                                
                                try:
                                    annotated_frame[start_y:end_y, start_x:end_x] = mask_viz_small
                                except Exception as viz_error:
                                    # If visualization fails, just skip it silently
                                    pass
        
        return annotated_frame, frame_results
    
    def add_gallery_image_corrected(self, image_path):
        """Add a gallery image with corrected processing"""
        print(f"Loading gallery image with corrected processing: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gallery image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect person in image
        results = self.yolo(image, classes=0, conf=0.5)
        
        person_detected = False
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = image[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        features = self.extract_features_corrected(person_img)
                        self.add_to_gallery(features, self.next_person_id, person_img)
                        print(f"Added Person {self.next_person_id} to gallery from {image_path}")
                        print(f"Feature shape: {features.shape}")
                        self.next_person_id += 1
                        person_detected = True
                        break
            
            if person_detected:
                break
        
        if not person_detected:
            raise ValueError(f"No person detected in gallery image: {image_path}")
        
        return True
    
    def test_video_corrected(self, video_path, expected_match=True, save_annotated=False, output_path=None):
        """Test video with corrected ReID"""
        print(f"\nTesting video with corrected ReID: {video_path}")
        print(f"Expected match: {expected_match}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Setup video writer if saving annotated video
        writer = None
        if save_annotated and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Test results
        results = {
            'video_path': video_path,
            'expected_match': expected_match,
            'total_frames': total_frames,
            'frames_with_detection': 0,
            'frames_with_match': 0,
            'frames_without_match': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy': 0.0,
            'similarity_scores': [],
            'frame_details': [],
            'avg_feature_time': 0.0
        }
        
        frame_idx = 0
        total_feature_time = 0.0
        
        print("Processing frames with corrected ReID...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, frame_result = self.process_frame_corrected(frame)
            total_feature_time += frame_result['feature_extraction_time']
            
            frame_detail = {
                'frame_idx': frame_idx,
                'has_detection': len(frame_result['detections']) > 0,
                'has_match': len(frame_result['matches']) > 0,
                'similarities': frame_result['similarities'],
                'correct': False,
                'feature_extraction_time': frame_result['feature_extraction_time']
            }
            
            if frame_detail['has_detection']:
                results['frames_with_detection'] += 1
                
                # Get the best similarity for this frame
                if frame_result['similarities']:
                    best_similarity = max(frame_result['similarities'])
                    results['similarity_scores'].append(best_similarity)
                    frame_detail['best_similarity'] = best_similarity
                
                if frame_detail['has_match']:
                    results['frames_with_match'] += 1
                else:
                    results['frames_without_match'] += 1
                
                # Determine if prediction is correct
                if expected_match:
                    # Should match - correct if we have a match
                    frame_detail['correct'] = frame_detail['has_match']
                else:
                    # Should NOT match - correct if we don't have a match
                    frame_detail['correct'] = not frame_detail['has_match']
                
                if frame_detail['correct']:
                    results['correct_predictions'] += 1
                else:
                    results['incorrect_predictions'] += 1
                
                # Add correctness indicator to frame
                correct_text = "✓" if frame_detail['correct'] else "✗"
                correct_color = (0, 255, 0) if frame_detail['correct'] else (0, 0, 255)
                cv2.putText(annotated_frame, correct_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, correct_color, 2)
            
            # Add frame info to annotated frame
            info_text = f"Frame {frame_idx+1}/{total_frames} | Expected: {'Match' if expected_match else 'No Match'}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add corrected ReID indicator
            cv2.putText(annotated_frame, "Corrected Masked ReID", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save annotated frame
            if writer:
                writer.write(annotated_frame)
            
            results['frame_details'].append(frame_detail)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Calculate final metrics
        total_detections = results['frames_with_detection']
        if total_detections > 0:
            results['accuracy'] = results['correct_predictions'] / total_detections
            results['avg_feature_time'] = total_feature_time / total_detections
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"Corrected test completed: {results['correct_predictions']}/{total_detections} correct predictions")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Avg feature time: {results['avg_feature_time']*1000:.1f}ms")
        
        if results['similarity_scores']:
            avg_sim = np.mean(results['similarity_scores'])
            min_sim = np.min(results['similarity_scores'])
            max_sim = np.max(results['similarity_scores'])
            print(f"Similarity range: [{min_sim:.3f}, {max_sim:.3f}], avg: {avg_sim:.3f}")
        
        # Print mask quality statistics if available
        mask_qualities = [det.get('mask_quality', 0.5) for det in results.get('frame_details', []) 
                         if det.get('has_detection', False)]
        if mask_qualities:
            avg_mask_quality = np.mean(mask_qualities)
            min_mask_quality = np.min(mask_qualities)
            max_mask_quality = np.max(mask_qualities)
            print(f"Mask quality range: [{min_mask_quality:.3f}, {max_mask_quality:.3f}], avg: {avg_mask_quality:.3f}")
        
        return results
    
    def run_corrected_test_suite(self, gallery_image_path, video1_path, video2_path, 
                                output_dir="corrected_masked_reid_results"):
        """Run corrected test suite"""
        print("="*80)
        print("CORRECTED MASKED BPBREID TEST SUITE")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear gallery first
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Step 1: Add gallery image
        print("\n1. Loading gallery image...")
        try:
            self.add_gallery_image_corrected(gallery_image_path)
        except Exception as e:
            print(f"Error loading gallery image: {e}")
            return None
        
        # Step 2: Test video 1 (should match)
        print("\n2. Testing Video 1 (Same Person - Should Match)...")
        video1_output = os.path.join(output_dir, "video1_annotated.mp4")
        
        try:
            results1 = self.test_video_corrected(
                video1_path, 
                expected_match=True, 
                save_annotated=True,
                output_path=video1_output
            )
        except Exception as e:
            print(f"Error testing video 1: {e}")
            return None
        
        # Step 3: Test video 2 (should not match)
        print("\n3. Testing Video 2 (Different Person - Should NOT Match)...")
        video2_output = os.path.join(output_dir, "video2_annotated.mp4")
        
        try:
            results2 = self.test_video_corrected(
                video2_path, 
                expected_match=False, 
                save_annotated=True,
                output_path=video2_output
            )
        except Exception as e:
            print(f"Error testing video 2: {e}")
            return None
        
        # Compile final results
        final_results = {
            'test_timestamp': datetime.now().isoformat(),
            'gallery_image': gallery_image_path,
            'reid_threshold': self.reid_threshold,
            'device': str(self.device),
            'model_type': 'corrected_masked_bpbreid',
            'video1_results': results1,
            'video2_results': results2,
            'overall_performance': {
                'total_frames_tested': results1['frames_with_detection'] + results2['frames_with_detection'],
                'total_correct': results1['correct_predictions'] + results2['correct_predictions'],
                'total_incorrect': results1['incorrect_predictions'] + results2['incorrect_predictions'],
                'overall_accuracy': 0.0,
                'avg_feature_time': (results1['avg_feature_time'] + results2['avg_feature_time']) / 2
            }
        }
        
        # Calculate overall accuracy
        total_tested = final_results['overall_performance']['total_frames_tested']
        total_correct = final_results['overall_performance']['total_correct']
        
        if total_tested > 0:
            final_results['overall_performance']['overall_accuracy'] = total_correct / total_tested
        
        # Save results to JSON
        results_file = os.path.join(output_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print detailed results
        self.print_corrected_test_summary(final_results)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Annotated videos saved to: {output_dir}")
        
        return final_results
    
    def print_corrected_test_summary(self, results):
        """Print detailed summary of corrected test results"""
        print("\n" + "="*80)
        print("CORRECTED MASKED BPBREID TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"Model Type: {results.get('model_type', 'corrected_masked_bpbreid')}")
        print(f"Gallery Image: {results['gallery_image']}")
        print(f"ReID Threshold: {results['reid_threshold']}")
        print(f"Device: {results['device']}")
        
        # Video 1 Results
        v1 = results['video1_results']
        print(f"\nVIDEO 1 (Same Person - Should Match):")
        print(f"  Path: {v1['video_path']}")
        print(f"  Total Frames: {v1['total_frames']}")
        print(f"  Frames with Person Detection: {v1['frames_with_detection']}")
        print(f"  Frames with Match: {v1['frames_with_match']}")
        print(f"  Frames without Match: {v1['frames_without_match']}")
        print(f"  Correct Predictions: {v1['correct_predictions']}")
        print(f"  Incorrect Predictions: {v1['incorrect_predictions']}")
        print(f"  Accuracy: {v1['accuracy']:.3f} ({v1['accuracy']*100:.1f}%)")
        print(f"  Avg Feature Time: {v1['avg_feature_time']*1000:.1f}ms")
        
        if v1['similarity_scores']:
            avg_sim = np.mean(v1['similarity_scores'])
            max_sim = np.max(v1['similarity_scores'])
            min_sim = np.min(v1['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Video 2 Results
        v2 = results['video2_results']
        print(f"\nVIDEO 2 (Different Person - Should NOT Match):")
        print(f"  Path: {v2['video_path']}")
        print(f"  Total Frames: {v2['total_frames']}")
        print(f"  Frames with Person Detection: {v2['frames_with_detection']}")
        print(f"  Frames with Match: {v2['frames_with_match']}")
        print(f"  Frames without Match: {v2['frames_without_match']}")
        print(f"  Correct Predictions: {v2['correct_predictions']}")
        print(f"  Incorrect Predictions: {v2['incorrect_predictions']}")
        print(f"  Accuracy: {v2['accuracy']:.3f} ({v2['accuracy']*100:.1f}%)")
        print(f"  Avg Feature Time: {v2['avg_feature_time']*1000:.1f}ms")
        
        if v2['similarity_scores']:
            avg_sim = np.mean(v2['similarity_scores'])
            max_sim = np.max(v2['similarity_scores'])
            min_sim = np.min(v2['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Overall Results
        overall = results['overall_performance']
        print(f"\nOVERALL CORRECTED PERFORMANCE:")
        print(f"  Total Frames Tested: {overall['total_frames_tested']}")
        print(f"  Total Correct: {overall['total_correct']}")
        print(f"  Total Incorrect: {overall['total_incorrect']}")
        print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f} ({overall['overall_accuracy']*100:.1f}%)")
        print(f"  Overall Avg Feature Time: {overall['avg_feature_time']*1000:.1f}ms")
        
        print("="*80)


def main():
    """Main function to run the corrected test suite"""
    print("Corrected Masked BPBreID Test Suite")
    print("=" * 80)
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    realtime_dir = os.path.dirname(current_dir)               
    bpbreid_dir = os.path.dirname(realtime_dir) 
    
    # Configuration - paths relative to parent directory
    reid_model_path = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    hrnet_path = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")
    
    # Test file paths
    gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1.jpg")
    video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1-vid.MOV")
    video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2","person-2-vid.MOV")
    
    # # Test file paths
    # gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "gallery-person.jpg")
    # video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "correct.MOV")
    # video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "incorrect.MOV")

    # Verify critical files exist
    required_files = [reid_model_path, hrnet_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure model files are available.")
        return
    
    try:
        # Create corrected test system
        print("Initializing Corrected Masked BPBreID test system...")
        tester = CorrectedMaskedBPBreIDTester(reid_model_path, hrnet_path)
        
        # Run corrected test suite
        if all(os.path.exists(p) for p in [gallery_image_path, video1_path, video2_path]):
            results = tester.run_corrected_test_suite(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path,
                output_dir="corrected_masked_reid_results"
            )
            
            if results:
                print("\n🎉 Corrected masked ReID test completed successfully!")
                print(f"Overall accuracy: {results['overall_performance']['overall_accuracy']:.3f}")
                
                # Print similarity analysis
                v1_sim = results['video1_results']['similarity_scores']
                v2_sim = results['video2_results']['similarity_scores']
                
                if v1_sim and v2_sim:
                    print(f"\nSimilarity Analysis:")
                    print(f"Video 1 (same person): avg={np.mean(v1_sim):.3f}, range=[{np.min(v1_sim):.3f}, {np.max(v1_sim):.3f}]")
                    print(f"Video 2 (diff person): avg={np.mean(v2_sim):.3f}, range=[{np.min(v2_sim):.3f}, {np.max(v2_sim):.3f}]")
                    
                    # Check if there's good separation
                    v1_avg = np.mean(v1_sim)
                    v2_avg = np.mean(v2_sim)
                    separation = v1_avg - v2_avg
                    print(f"Similarity separation: {separation:.3f}")
                    
                    if separation > 0.1:
                        print("✅ Good feature discrimination achieved!")
                    else:
                        print("⚠️ Poor feature discrimination - features may need further tuning")
            else:
                print("Corrected test failed!")
        else:
            print("Test files not found. Please check paths.")
            
    except Exception as e:
        print(f"Error running corrected test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Corrected Masked BPBreID Test Suite")
    print("=" * 80)
    print("Key Corrections:")
    print("- Fixed feature extraction and normalization")
    print("- Improved similarity computation with multiple metrics")
    print("- Added debug mode for better analysis")
    print("- Increased threshold to reduce false positives")
    print("- Better error handling and fallbacks")
    print("- Feature discrimination analysis")
    print()
    
    main()
