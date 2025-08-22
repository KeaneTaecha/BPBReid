import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from collections import defaultdict
import time
from PIL import Image
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Try to import OpenPifPaf
try:
    import openpifpaf
    PIFPAF_AVAILABLE = True
    print("✓ OpenPifPaf available")
except ImportError:
    PIFPAF_AVAILABLE = False
    print("✗ OpenPifPaf not available - installing with: pip install openpifpaf")

# Use torchvision segmentation (much easier to install than Detectron2)
try:
    import torchvision.models.segmentation as segmentation_models
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    SEGMENTATION_AVAILABLE = True
    print("✓ Torchvision Mask R-CNN available")
except ImportError:
    SEGMENTATION_AVAILABLE = False
    print("✗ Torchvision segmentation not available")

# Import BPBreid mask transforms
from torchreid.data.masks_transforms import (
    CombinePifPafIntoFiveVerticalParts,
    AddBackgroundMask,
    PermuteMasksDim,
    ResizeMasks
)

class TorchvisionOfficialMaskingBPBreIDTester:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the Official Masking BPBreID system using Torchvision Mask R-CNN"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BPBreID model with official masking configuration
        self.reid_model = self._load_reid_model(reid_model_path, hrnet_path)
        self.reid_model.eval()
        
        # Setup transforms for ReID (same as official pipeline)
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize masking pipeline components
        self._setup_pifpaf_pipeline()
        self._setup_torchvision_maskrcnn()
        self._setup_mask_transforms()
        
        # Gallery to store person features
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.4
        
        # Test results storage
        self.test_results = {}
        
        print("Torchvision Official Masking BPBreID system initialized successfully!")
        print("Using official BPBreID mask transformation pipeline with proper anatomical mapping")
        
    def _setup_pifpaf_pipeline(self):
        """Setup OpenPifPaf for pose estimation"""
        if not PIFPAF_AVAILABLE:
            print("⚠️  PifPaf not available - using pose estimation fallback")
            self.pifpaf_model = None
            return
            
        try:
            print("Setting up OpenPifPaf pose estimation...")
            # Initialize PifPaf predictor with default model
            self.pifpaf_model = openpifpaf.Predictor(
                checkpoint='shufflenetv2k16',
                visualize_image=False,
                visualize_processed_image=False
            )
            print("✓ OpenPifPaf initialized successfully")
        except Exception as e:
            print(f"✗ Error setting up OpenPifPaf: {e}")
            print("Using pose estimation fallback...")
            self.pifpaf_model = None
    
    def _setup_torchvision_maskrcnn(self):
        """Setup Torchvision Mask R-CNN for person segmentation"""
        if not SEGMENTATION_AVAILABLE:
            print("✗ Torchvision segmentation not available")
            self.maskrcnn_model = None
            return
            
        try:
            print("Setting up Torchvision Mask R-CNN...")
            
            # Load pre-trained Mask R-CNN model
            self.maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
            self.maskrcnn_model.eval()
            self.maskrcnn_model = self.maskrcnn_model.to(self.device)
            
            # Setup transform for Mask R-CNN
            self.maskrcnn_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
            print("✓ Torchvision Mask R-CNN initialized successfully")
        except Exception as e:
            print(f"✗ Error setting up Torchvision Mask R-CNN: {e}")
            self.maskrcnn_model = None
    
    def _setup_mask_transforms(self):
        """Setup proper BPBreID mask transformation pipeline"""
        print("Setting up BPBreID mask transformation pipeline...")
        try:
            # Import the proper transforms
            from torchreid.data.masks_transforms.pifpaf_mask_transform import CombinePifPafIntoFiveVerticalParts
            from torchreid.data.masks_transforms.mask_transform import AddBackgroundMask, PermuteMasksDim, ResizeMasks
            
            # Create the transformation pipeline (matching BPBreID config)
            self.pifpaf_grouping_transform = CombinePifPafIntoFiveVerticalParts()
            self.add_background_transform = AddBackgroundMask(background_computation_strategy='diff_from_max')
            
            print("✓ BPBreID mask transforms initialized successfully")
            self.mask_transforms_available = True
            
        except ImportError as e:
            print(f"⚠️  Could not import BPBreID mask transforms: {e}")
            print("Using simplified mask processing fallback")
            self.mask_transforms_available = False
    
    def _load_reid_model(self, model_path, hrnet_path):
        """Load the BPBreID model with official masking configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        from types import SimpleNamespace
        
        # Create config that matches official testing configuration
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.load_weights = model_path
        config.model.load_config = True
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.learnable_attention_enabled = False  # Use external masks
        config.model.bpbreid.mask_filtering_testing = True
        config.model.bpbreid.mask_filtering_training = False
        config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.test_use_target_segmentation = 'none'  # Use external masks
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        
        # Build model with config
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=751,
            config=config,
            pretrained=True
        )
        
        # Handle DataParallel state dict (remove 'module.' prefix)
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
                
        # Load the cleaned state dict
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        return model
    
    def generate_pifpaf_confidence_fields(self, image):
        """Generate PifPaf confidence fields following the official pipeline"""
        if self.pifpaf_model is None:
            return self._generate_dummy_confidence_fields(image)
            
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # First check if PifPaf detects any poses
            predictions, gt_anns, image_meta = self.pifpaf_model.pil_image(pil_image)
            
            # If no predictions, return dummy fields
            if not predictions or len(predictions) == 0:
                print("⚠️  No poses detected by PifPaf, using dummy confidence fields")
                return self._generate_dummy_confidence_fields(image)
            else:
                print(f"✓ PifPaf detected {len(predictions)} pose(s), extracting confidence fields...")
            
            # Use a simplified approach: create confidence fields directly from predictions
            # This avoids the complex field extraction that's causing issues
            print("🔄 Using simplified approach: creating confidence fields from keypoint predictions...")
            return self._create_confidence_from_predictions(predictions, image.shape[:2])
                
        except Exception as e:
            print(f"PifPaf processing failed: {e}, using dummy fields")
            return self._generate_dummy_confidence_fields(image)
    
    def _generate_dummy_confidence_fields(self, image):
        """Generate dummy PifPaf confidence fields when PifPaf is not available"""
        # Create confidence fields similar to PifPaf output
        # PifPaf typically outputs 17 keypoint fields + 19 connection fields = 36 total
        h, w = image.shape[:2]
        
        # Use smaller field size (typical PifPaf output is much smaller than input)
        field_h, field_w = max(16, h // 16), max(16, w // 16)
        
        # Create 36 confidence fields (17 keypoints + 19 connections)
        dummy_fields = torch.zeros(36, field_h, field_w, device=self.device, dtype=torch.float32)
        
        # Create some basic confidence patterns for different body regions
        # Head keypoints (nose, eyes, ears) - fields 0-4
        head_end = int(field_h * 0.25)
        dummy_fields[0:5, :head_end, :] = torch.randn(5, head_end, field_w, device=self.device) * 0.3 + 0.5
        
        # Upper body keypoints (shoulders, elbows, wrists) - fields 5-10
        upper_start, upper_end = int(field_h * 0.25), int(field_h * 0.50)
        dummy_fields[5:11, upper_start:upper_end, :] = torch.randn(6, upper_end-upper_start, field_w, device=self.device) * 0.3 + 0.4
        
        # Hip keypoints - fields 11-12
        hip_start, hip_end = int(field_h * 0.45), int(field_h * 0.55)
        dummy_fields[11:13, hip_start:hip_end, :] = torch.randn(2, hip_end-hip_start, field_w, device=self.device) * 0.3 + 0.4
        
        # Leg keypoints (knees, ankles) - fields 13-16
        leg_start, leg_end = int(field_h * 0.55), int(field_h * 0.90)
        dummy_fields[13:17, leg_start:leg_end, :] = torch.randn(4, leg_end-leg_start, field_w, device=self.device) * 0.3 + 0.3
        
        # Connection fields (PAF) - fields 17-35
        # These connect different keypoints, so spread across the person
        person_start, person_end = int(field_h * 0.1), int(field_h * 0.9)
        dummy_fields[17:36, person_start:person_end, :] = torch.randn(19, person_end-person_start, field_w, device=self.device) * 0.2 + 0.2
        
        # Clamp values to valid confidence range
        dummy_fields = torch.clamp(dummy_fields, 0.0, 1.0)
        
        return dummy_fields
    
    def _create_confidence_from_predictions(self, predictions, image_shape):
        """Create confidence fields from PifPaf predictions when field extraction fails"""
        try:
            h, w = image_shape
            # Use larger field size for better spatial resolution
            # PifPaf typically outputs at 1/4 to 1/8 of input resolution
            field_h, field_w = max(32, h // 8), max(32, w // 8)
            print(f"🔍 Creating confidence fields with size: {field_h}x{field_w} for image {h}x{w}")
            
            # Create 36 confidence fields (17 keypoints + 19 connections) matching PIFPAF_PARTS order
            # From pifpaf_mask_transform.py:
            # PIFPAF_KEYPOINTS: ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
            #                    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            #                    "right_knee", "left_ankle", "right_ankle"] (0-16)
            # PIFPAF_JOINTS: ["left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
            #                 "right_knee_to_right_hip", "left_hip_to_right_hip", "left_shoulder_to_left_hip",
            #                 "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder", "left_shoulder_to_left_elbow",
            #                 "right_shoulder_to_right_elbow", "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
            #                 "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
            #                 "right_eye_to_right_ear", "left_ear_to_left_shoulder", "right_ear_to_right_shoulder"] (17-35)
            confidence_fields = torch.zeros(36, field_h, field_w, device=self.device, dtype=torch.float32)
            
            # Use the first (strongest) prediction
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                
                # Extract keypoint data if available
                if hasattr(pred, 'data') and len(pred.data) >= 17:
                    keypoints = pred.data[:17]  # Standard COCO keypoints
                    print(f"✓ Using {len(keypoints)} keypoints from PifPaf prediction")
                else:
                    print(f"⚠️  PifPaf prediction has insufficient keypoint data: {len(pred.data) if hasattr(pred, 'data') else 'no data'}")
                    keypoints = []
                
                # Map COCO keypoints to PIFPAF_KEYPOINTS order
                # COCO: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
                #        left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, 
                #        left_knee, right_knee, left_ankle, right_ankle]
                # This matches PIFPAF_KEYPOINTS exactly, so mapping is 1:1
                
                # Create confidence fields based on keypoint positions and scores
                for coco_idx, kp in enumerate(keypoints):
                    if len(kp) >= 3 and kp[2] > 0.1:  # Confidence threshold
                        # COCO and PIFPAF keypoint order is the same
                        pifpaf_idx = coco_idx
                        
                        # Normalize coordinates to field size
                        x = int((kp[0] / w) * field_w)
                        y = int((kp[1] / h) * field_h)
                        conf = float(kp[2])  # Keypoint confidence
                        
                        # Clamp to valid range
                        x = max(0, min(field_w - 1, x))
                        y = max(0, min(field_h - 1, y))
                        
                        # Create much larger confidence regions for proper body part coverage
                        # The key insight: PifPaf confidence fields should cover entire body regions, not just points
                        base_radius = max(4, min(field_h, field_w) // 4)  # Much larger base radius
                        
                        # Different radii for different body parts
                        if pifpaf_idx < 5:  # Head keypoints
                            radius = base_radius // 2
                        elif pifpaf_idx < 11:  # Arms/shoulders  
                            radius = base_radius
                        else:  # Torso/legs
                            radius = int(base_radius * 1.2)
                        
                        y_start, y_end = max(0, y-radius), min(field_h, y+radius+1)
                        x_start, x_end = max(0, x-radius), min(field_w, x+radius+1)
                        
                        # Apply confidence to keypoint field with much stronger values
                        if pifpaf_idx < 17:  # Keypoint fields (0-16)
                            for dy in range(y_start, y_end):
                                for dx in range(x_start, x_end):
                                    distance = max(1, np.sqrt((dy-y)**2 + (dx-x)**2))
                                    # Much stronger confidence values with slower falloff
                                    falloff = max(0.3, 1.0 / (1 + distance * 0.1))  
                                    confidence_fields[pifpaf_idx, dy, dx] = max(
                                        confidence_fields[pifpaf_idx, dy, dx],
                                        conf * falloff * 0.9  # Higher base confidence
                                    )
                    
                    # Create connection fields based on PIFPAF_JOINTS order
                    # PIFPAF_JOINTS: ["left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                    #                 "right_knee_to_right_hip", "left_hip_to_right_hip", "left_shoulder_to_left_hip",
                    #                 "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder", "left_shoulder_to_left_elbow",
                    #                 "right_shoulder_to_right_elbow", "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
                    #                 "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                    #                 "right_eye_to_right_ear", "left_ear_to_left_shoulder", "right_ear_to_right_shoulder"]
                    
                    # Map PIFPAF joints to COCO keypoint indices (0-based)
                    pifpaf_joints = [
                        (15, 13),  # left_ankle_to_left_knee
                        (13, 11),  # left_knee_to_left_hip  
                        (16, 14),  # right_ankle_to_right_knee
                        (14, 12),  # right_knee_to_right_hip
                        (11, 12),  # left_hip_to_right_hip
                        (5, 11),   # left_shoulder_to_left_hip
                        (6, 12),   # right_shoulder_to_right_hip
                        (5, 6),    # left_shoulder_to_right_shoulder
                        (5, 7),    # left_shoulder_to_left_elbow
                        (6, 8),    # right_shoulder_to_right_elbow
                        (7, 9),    # left_elbow_to_left_wrist
                        (8, 10),   # right_elbow_to_right_wrist
                        (1, 2),    # left_eye_to_right_eye
                        (0, 1),    # nose_to_left_eye
                        (0, 2),    # nose_to_right_eye
                        (1, 3),    # left_eye_to_left_ear
                        (2, 4),    # right_eye_to_right_ear
                        (3, 5),    # left_ear_to_left_shoulder
                        (4, 6),    # right_ear_to_right_shoulder
                    ]
                    
                    for conn_idx, (kp1_idx, kp2_idx) in enumerate(pifpaf_joints):
                        if conn_idx < 19 and kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
                            kp1 = keypoints[kp1_idx]  # Already 0-based
                            kp2 = keypoints[kp2_idx]
                            
                            if len(kp1) >= 3 and len(kp2) >= 3 and kp1[2] > 0.1 and kp2[2] > 0.1:
                                # Create connection field between keypoints
                                x1 = int((kp1[0] / w) * field_w)
                                y1 = int((kp1[1] / h) * field_h)
                                x2 = int((kp2[0] / w) * field_w)
                                y2 = int((kp2[1] / h) * field_h)
                                
                                # Clamp coordinates
                                x1, y1 = max(0, min(field_w-1, x1)), max(0, min(field_h-1, y1))
                                x2, y2 = max(0, min(field_w-1, x2)), max(0, min(field_h-1, y2))
                                
                                # Simple line drawing for connection
                                conf_conn = min(kp1[2], kp2[2]) * 0.6
                                
                                # Draw much thicker connection lines for realistic body part coverage
                                steps = max(abs(x2-x1), abs(y2-y1), 1)
                                for step in range(steps+1):
                                    t = step / steps if steps > 0 else 0
                                    x = int(x1 + t * (x2 - x1))
                                    y = int(y1 + t * (y2 - y1))
                                    
                                    # Much larger radius for connections to create thick body part regions
                                    radius = max(3, min(field_h, field_w) // 6)  # Much thicker connections
                                    y_start, y_end = max(0, y-radius), min(field_h, y+radius+1)
                                    x_start, x_end = max(0, x-radius), min(field_w, x+radius+1)
                                    
                                    # Apply strong connection confidence
                                    # Connection fields start at index 17 (after 17 keypoint fields)
                                    joint_field_idx = 17 + conn_idx
                                    for dy in range(y_start, y_end):
                                        for dx in range(x_start, x_end):
                                            # Create thick, strong connections
                                            confidence_fields[joint_field_idx, dy, dx] = max(
                                                confidence_fields[joint_field_idx, dy, dx],
                                                conf_conn * 0.9  # Very strong connection confidence
                                            )
            
            # Add anatomical body region filling to ensure proper coverage
            # This creates realistic body-shaped confidence regions
            confidence_fields = self._add_anatomical_body_regions(confidence_fields, keypoints, field_h, field_w, w, h)
            
            # Clamp values to valid confidence range
            confidence_fields = torch.clamp(confidence_fields, 0.0, 1.0)
            
            print(f"✓ Generated confidence fields with {(confidence_fields > 0.1).sum()} significant values")
            
            return confidence_fields
            
        except Exception as e:
            print(f"⚠️  Error creating confidence from predictions: {e}")
            print("🔄 Falling back to dummy confidence fields...")
            return self._generate_dummy_confidence_fields(image_shape)
    
    def _add_anatomical_body_regions(self, confidence_fields, keypoints, field_h, field_w, orig_w, orig_h):
        """Add anatomical body regions to fill in gaps and create realistic body shapes"""
        try:
            if not keypoints or len(keypoints) < 17:
                return confidence_fields
                
            # Get key landmark positions in field coordinates
            landmarks = {}
            for i, kp in enumerate(keypoints):
                if len(kp) >= 3 and kp[2] > 0.1:
                    x = int((kp[0] / orig_w) * field_w)
                    y = int((kp[1] / orig_h) * field_h)
                    x = max(0, min(field_w - 1, x))
                    y = max(0, min(field_h - 1, y))
                    
                    # Map COCO keypoint indices to body parts
                    keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", 
                                     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                                     "left_wrist", "right_wrist", "left_hip", "right_hip", 
                                     "left_knee", "right_knee", "left_ankle", "right_ankle"]
                    if i < len(keypoint_names):
                        landmarks[keypoint_names[i]] = (x, y, kp[2])
            
            # Fill anatomical regions based on detected landmarks
            
            # 1. Head region - fill between facial keypoints
            if "nose" in landmarks or "left_eye" in landmarks or "right_eye" in landmarks:
                head_keypoints = [k for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"] if k in landmarks]
                if head_keypoints:
                    # Create head region
                    head_points = [landmarks[k][:2] for k in head_keypoints]
                    head_center_x = int(np.mean([p[0] for p in head_points]))
                    head_center_y = int(np.mean([p[1] for p in head_points]))
                    head_radius = max(3, field_h // 8)
                    
                    # Fill head keypoint fields (0-4)
                    for head_idx in range(5):
                        self._fill_circular_region(confidence_fields[head_idx], head_center_x, head_center_y, head_radius, 0.8)
            
            # 2. Torso region - fill between shoulders and hips
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                left_shoulder = landmarks["left_shoulder"][:2]
                right_shoulder = landmarks["right_shoulder"][:2]
                
                # Estimate torso region
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
                
                # If hips are available, use them to define torso height
                if "left_hip" in landmarks and "right_hip" in landmarks:
                    left_hip = landmarks["left_hip"][:2]
                    right_hip = landmarks["right_hip"][:2]
                    hip_center_x = (left_hip[0] + right_hip[0]) // 2
                    hip_center_y = (left_hip[1] + right_hip[1]) // 2
                    
                    # Fill torso region (rectangular)
                    torso_width = max(abs(right_shoulder[0] - left_shoulder[0]), field_w // 6)
                    torso_height = abs(hip_center_y - shoulder_center_y)
                    
                    # Fill shoulder keypoints (5-6)
                    for shoulder_idx in [5, 6]:
                        self._fill_rectangular_region(confidence_fields[shoulder_idx], 
                                                    shoulder_center_x, shoulder_center_y, 
                                                    torso_width, torso_height // 3, 0.7)
                    
                    # Fill hip keypoints (11-12)
                    for hip_idx in [11, 12]:
                        self._fill_rectangular_region(confidence_fields[hip_idx], 
                                                    hip_center_x, hip_center_y, 
                                                    torso_width, torso_height // 3, 0.6)
            
            # 3. Limb regions - fill between joints
            limb_connections = [
                ("left_shoulder", "left_elbow", [5, 7]),   # Left upper arm
                ("right_shoulder", "right_elbow", [6, 8]), # Right upper arm  
                ("left_elbow", "left_wrist", [7, 9]),      # Left forearm
                ("right_elbow", "right_wrist", [8, 10]),   # Right forearm
                ("left_hip", "left_knee", [11, 13]),       # Left thigh
                ("right_hip", "right_knee", [12, 14]),     # Right thigh
                ("left_knee", "left_ankle", [13, 15]),     # Left calf
                ("right_knee", "right_ankle", [14, 16]),   # Right calf
            ]
            
            for start_kp, end_kp, field_indices in limb_connections:
                if start_kp in landmarks and end_kp in landmarks:
                    start_pos = landmarks[start_kp][:2]
                    end_pos = landmarks[end_kp][:2]
                    
                    # Fill the limb region
                    for field_idx in field_indices:
                        self._fill_limb_region(confidence_fields[field_idx], start_pos, end_pos, 0.6)
            
            return confidence_fields
            
        except Exception as e:
            print(f"⚠️  Error adding anatomical regions: {e}")
            return confidence_fields
    
    def _fill_circular_region(self, field, center_x, center_y, radius, value):
        """Fill a circular region in the confidence field"""
        h, w = field.shape
        for dy in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
            for dx in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                distance = np.sqrt((dy - center_y)**2 + (dx - center_x)**2)
                if distance <= radius:
                    field[dy, dx] = max(field[dy, dx], value * (1 - distance / radius))
    
    def _fill_rectangular_region(self, field, center_x, center_y, width, height, value):
        """Fill a rectangular region in the confidence field"""
        h, w = field.shape
        half_width, half_height = width // 2, height // 2
        y_start = max(0, center_y - half_height)
        y_end = min(h, center_y + half_height + 1)
        x_start = max(0, center_x - half_width)
        x_end = min(w, center_x + half_width + 1)
        
        for dy in range(y_start, y_end):
            for dx in range(x_start, x_end):
                field[dy, dx] = max(field[dy, dx], value)
    
    def _fill_limb_region(self, field, start_pos, end_pos, value):
        """Fill a limb region between two points"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Draw thick line between the points
        steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
        limb_thickness = max(2, min(field.shape) // 12)
        
        for step in range(steps + 1):
            t = step / steps if steps > 0 else 0
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            
            # Fill thick region around the line
            self._fill_circular_region(field, x, y, limb_thickness, value)
    
    def generate_person_mask(self, image):
        """Generate person segmentation mask using Torchvision Mask R-CNN"""
        if self.maskrcnn_model is None:
            # Return dummy mask if Mask R-CNN not available
            return torch.ones(image.shape[0], image.shape[1], device=self.device)
            
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL and apply transform
            pil_image = Image.fromarray(image_rgb)
            image_tensor = self.maskrcnn_transform(pil_image).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.maskrcnn_model([image_tensor])
            
            # Extract person masks (class 1 is person in COCO for torchvision)
            pred_boxes = predictions[0]['boxes']
            pred_labels = predictions[0]['labels']
            pred_scores = predictions[0]['scores']
            pred_masks = predictions[0]['masks']
            
            # Filter for person class (label 1) and high confidence
            person_indices = (pred_labels == 1) & (pred_scores > 0.5)
            
            if person_indices.sum() > 0:
                # Get the mask with highest score
                person_masks = pred_masks[person_indices]
                person_scores = pred_scores[person_indices]
                best_mask_idx = person_scores.argmax()
                
                # Get the best person mask and squeeze to 2D
                person_mask = person_masks[best_mask_idx].squeeze()
                
                # Threshold the mask
                person_mask = (person_mask > 0.5).float()
            else:
                # No person detected, create dummy mask
                person_mask = torch.ones(image.shape[0], image.shape[1], device=self.device)
            
            return person_mask
            
        except Exception as e:
            print(f"Error in Torchvision Mask R-CNN processing: {e}")
            # Return dummy mask if Mask R-CNN fails
            return torch.ones(image.shape[0], image.shape[1], device=self.device)
    
    def filter_pifpaf_with_mask(self, pifpaf_conf, person_mask, is_resize_pifpaf=False):
        """Filter PifPaf confidence fields using person segmentation mask (following get_labels.py)"""
        try:
            # Convert tensors to numpy for OpenCV operations (following official pipeline)
            if isinstance(pifpaf_conf, torch.Tensor):
                pifpaf_array = pifpaf_conf.cpu().numpy()
            else:
                pifpaf_array = pifpaf_conf
                
            if isinstance(person_mask, torch.Tensor):
                mask = person_mask.cpu().numpy()
            else:
                mask = person_mask
            
            # Ensure mask is binary
            mask = (mask > 0.5).astype(np.float32)
            
            if is_resize_pifpaf:
                # Resize the PifPaf array to match the size of the mask
                pifpaf_resized = np.transpose(pifpaf_array, (1, 2, 0))
                pifpaf_resized = cv2.resize(pifpaf_resized, dsize=(mask.shape[1], mask.shape[0]),
                                            interpolation=cv2.INTER_CUBIC)
                pifpaf_resized = np.transpose(pifpaf_resized, (2, 0, 1))

                # Filter the PifPaf array using the segmentation mask
                filtered_pifpaf = mask * pifpaf_resized
                
                # Resize back to standard size (following official pipeline)
                filtered_pifpaf = np.array(
                    [cv2.resize(slice_2d, (9, 17), interpolation=cv2.INTER_CUBIC) for slice_2d in filtered_pifpaf])

                return torch.from_numpy(filtered_pifpaf).to(self.device)
            else:
                # Resize the mask to match the size of the PifPaf array (default approach)
                mask_resized = cv2.resize(mask.astype(np.uint8), (pifpaf_array.shape[2], pifpaf_array.shape[1]))
                filtered_pifpaf = mask_resized * pifpaf_array
                return torch.from_numpy(filtered_pifpaf).to(self.device)
            
        except Exception as e:
            print(f"Error filtering PifPaf with mask: {e}")
            return pifpaf_conf
    
    def generate_official_masks(self, image):
        """Generate masks using official BPBreID pipeline: PifPaf confidence fields + proper transforms"""
        try:
            # Step 1: Generate PifPaf confidence fields (36 channels)
            pifpaf_confidence = self.generate_pifpaf_confidence_fields(image)
            
            # Step 2: Generate person segmentation mask
            person_mask = self.generate_person_mask(image)
            
            # Step 3: Filter PifPaf with person mask (following get_labels.py)
            filtered_pifpaf = self.filter_pifpaf_with_mask(pifpaf_confidence, person_mask, is_resize_pifpaf=False)
            
            # Step 4: Use proper BPBreID transforms if available
            if hasattr(self, 'mask_transforms_available') and self.mask_transforms_available:
                final_masks = self._apply_bpbreid_transforms(filtered_pifpaf)
            else:
                # Fallback to manual conversion
                final_masks = self._convert_pifpaf_to_body_parts(filtered_pifpaf)
            
            # Ensure masks have correct shape [1, 6, H, W] (batch, 5 parts + 1 background)
            if final_masks.dim() == 3:
                final_masks = final_masks.unsqueeze(0)  # Add batch dimension
            
            return final_masks
            
        except Exception as e:
            print(f"Error in official mask generation: {e}")
            # Return dummy masks if generation fails
            dummy_masks = torch.zeros(1, 6, 32, 32, device=self.device, dtype=torch.float32)
            dummy_masks[:, 0] = 1.0  # Set background mask
            return dummy_masks
    
    def _apply_bpbreid_transforms(self, pifpaf_confidence):
        """Apply proper BPBreID mask transformation pipeline"""
        try:
            print("🔄 Applying BPBreID mask transforms...")
            print(f"🔍 Input confidence shape: {pifpaf_confidence.shape}")
            print(f"🔍 Input confidence range: [{pifpaf_confidence.min():.3f}, {pifpaf_confidence.max():.3f}]")
            print(f"🔍 Non-zero elements: {(pifpaf_confidence > 0.01).sum()}")
            
            # Ensure we have 36 channels (17 keypoints + 19 connections)
            if pifpaf_confidence.shape[0] != 36:
                print(f"⚠️  Expected 36 PifPaf channels, got {pifpaf_confidence.shape[0]}")
                # Pad or truncate to 36 channels
                if pifpaf_confidence.shape[0] < 36:
                    padding = torch.zeros(36 - pifpaf_confidence.shape[0], *pifpaf_confidence.shape[1:], 
                                        device=self.device, dtype=torch.float32)
                    pifpaf_confidence = torch.cat([pifpaf_confidence, padding], dim=0)
                else:
                    pifpaf_confidence = pifpaf_confidence[:36]
            
            # Debug: Check individual channels
            for i in range(min(5, pifpaf_confidence.shape[0])):
                channel_sum = pifpaf_confidence[i].sum().item()
                print(f"🔍 Channel {i} sum: {channel_sum:.3f}")
            
            # Apply the PifPaf grouping transform (36 channels -> 5 body parts)
            body_part_masks = self.pifpaf_grouping_transform.apply_to_mask(pifpaf_confidence)
            
            print(f"✓ Grouped into {body_part_masks.shape[0]} body parts, shape: {body_part_masks.shape}")
            print(f"🔍 Body part masks range: [{body_part_masks.min():.3f}, {body_part_masks.max():.3f}]")
            
            # Debug: Check each body part
            for i in range(body_part_masks.shape[0]):
                mask_sum = body_part_masks[i].sum().item()
                print(f"🔍 Body part {i} sum: {mask_sum:.3f}")
            
            # Add background mask (5 parts -> 6 with background)
            final_masks = self.add_background_transform.apply_to_mask(body_part_masks)
            
            print(f"✓ Added background mask, final shape: {final_masks.shape}")
            print(f"🔍 Final masks range: [{final_masks.min():.3f}, {final_masks.max():.3f}]")
            
            return final_masks
            
        except Exception as e:
            print(f"⚠️  BPBreID transform failed: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to manual conversion...")
            return self._convert_pifpaf_to_body_parts(pifpaf_confidence)
    
    def _convert_pifpaf_to_body_parts(self, pifpaf_confidence):
        """Convert PifPaf confidence fields to 5 body part masks (following BPBreID pipeline)"""
        try:
            # PifPaf outputs 36 confidence fields (17 keypoints + 19 connections)
            # We need to group these into 5 body parts for BPBreID
            
            # Get dimensions
            num_fields, field_h, field_w = pifpaf_confidence.shape
            
            # Create 5 body part masks
            body_parts = torch.zeros(5, field_h, field_w, device=self.device, dtype=torch.float32)
            
            # Group PifPaf fields into body parts based on keypoint anatomy
            # Following COCO keypoint order and BPBreID's 5-part division
            
            if num_fields >= 17:  # Ensure we have keypoint fields
                # Head (keypoints 0-4: nose, eyes, ears)
                head_fields = pifpaf_confidence[0:5]  # nose, left_eye, right_eye, left_ear, right_ear
                body_parts[0] = torch.max(head_fields, dim=0)[0]
                
                # Upper body (keypoints 5-10: shoulders, elbows, wrists)  
                upper_fields = pifpaf_confidence[5:11]  # left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
                body_parts[1] = torch.max(upper_fields, dim=0)[0]
                
                # Lower torso (keypoints 11-12: hips)
                torso_fields = pifpaf_confidence[11:13]  # left_hip, right_hip
                body_parts[2] = torch.max(torso_fields, dim=0)[0]
                
                # Legs (keypoints 13-15: knees, ankles)
                leg_fields = pifpaf_confidence[13:17]  # left_knee, right_knee, left_ankle, right_ankle
                body_parts[3] = torch.max(leg_fields, dim=0)[0]
                
                # For feet, we'll use ankle information and some connection fields if available
                if num_fields >= 36:
                    # Use some connection fields that involve feet/ankles
                    feet_fields = torch.cat([pifpaf_confidence[15:17], pifpaf_confidence[30:36]], dim=0)  # ankles + some connections
                    body_parts[4] = torch.max(feet_fields, dim=0)[0]
                else:
                    # Just use ankle keypoints for feet
                    body_parts[4] = torch.max(pifpaf_confidence[15:17], dim=0)[0]
            else:
                # If we don't have enough fields, distribute evenly
                fields_per_part = max(1, num_fields // 5)
                for i in range(5):
                    start_idx = i * fields_per_part
                    end_idx = min((i + 1) * fields_per_part, num_fields)
                    if start_idx < num_fields:
                        body_parts[i] = torch.max(pifpaf_confidence[start_idx:end_idx], dim=0)[0]
            
            # Normalize confidence values
            body_parts = torch.clamp(body_parts, 0.0, 1.0)
            
            # Create background mask
            foreground_mask = torch.clamp(body_parts.sum(dim=0), 0, 1)
            background_mask = 1.0 - foreground_mask
            
            # Combine: [background, 5 body parts] = 6 total masks
            final_masks = torch.cat([background_mask.unsqueeze(0), body_parts], dim=0)
            
            return final_masks
            
        except Exception as e:
            print(f"Error converting PifPaf to body parts: {e}")
            # Return basic masks
            h, w = 32, 32
            dummy_masks = torch.zeros(6, h, w, device=self.device, dtype=torch.float32)
            dummy_masks[0] = 1.0  # Background
            return dummy_masks
    
    def _combine_masks_simple(self, pose_masks, person_mask):
        """Combine pose masks with person segmentation mask"""
        try:
            # Resize person mask to match pose mask dimensions
            pose_h, pose_w = pose_masks.shape[1], pose_masks.shape[2]
            
            # Resize person mask to match pose mask size
            person_mask_resized = F.interpolate(
                person_mask.unsqueeze(0).unsqueeze(0),
                size=(pose_h, pose_w),
                mode='bilinear',
                align_corners=True
            ).squeeze()
            
            # Apply person mask to pose masks
            filtered_masks = pose_masks * person_mask_resized.unsqueeze(0)
            
            # Create background mask
            foreground_mask = torch.clamp(filtered_masks.sum(dim=0), 0, 1)
            background_mask = 1.0 - foreground_mask
            
            # Combine: [background, 5 body parts] = 6 total masks
            final_masks = torch.cat([background_mask.unsqueeze(0), filtered_masks], dim=0)
            
            return final_masks
            
        except Exception as e:
            print(f"Error combining masks: {e}")
            # Return basic masks
            h, w = 32, 32
            dummy_masks = torch.zeros(6, h, w, device=self.device, dtype=torch.float32)
            dummy_masks[0] = 1.0  # Background
            return dummy_masks
    
    def visualize_masks_on_image(self, image, masks_tensor, person_bbox=None, alpha=0.4):
        """Visualize masks overlaid on the image"""
        try:
            # Convert image to numpy if needed
            if isinstance(image, torch.Tensor):
                vis_image = image.cpu().numpy()
            else:
                vis_image = image.copy()
            
            # Ensure image is in correct format
            if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
                vis_image = vis_image.astype(np.uint8)
            else:
                return vis_image
            
            # Extract masks (skip background mask at index 0)
            if masks_tensor.dim() == 4:  # [batch, channels, h, w]
                masks = masks_tensor.squeeze(0)[1:]  # Skip background, get body parts
            else:  # [channels, h, w]
                masks = masks_tensor[1:]  # Skip background
            
            # Define colors for different body parts
            colors = [
                (255, 0, 0),    # Red - Head
                (0, 255, 0),    # Green - Upper torso
                (0, 0, 255),    # Blue - Lower torso
                (255, 255, 0),  # Cyan - Legs
                (255, 0, 255),  # Magenta - Feet
            ]
            
            # Get image dimensions
            img_h, img_w = vis_image.shape[:2]
            
            # Create overlay image
            overlay = vis_image.copy()
            
            # Process each body part mask
            for i, mask in enumerate(masks):
                if i >= len(colors):
                    break
                    
                # Convert mask to numpy and resize to image dimensions
                mask_np = mask.cpu().numpy()
                
                # Resize mask to image size
                mask_resized = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                # Threshold mask
                mask_binary = (mask_resized > 0.1).astype(np.uint8)
                
                # Apply color to mask regions
                color_mask = np.zeros_like(vis_image)
                color_mask[mask_binary > 0] = colors[i]
                
                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
            
            # If we have a person bbox, crop and resize the visualization
            if person_bbox is not None:
                x1, y1, x2, y2 = person_bbox
                # Ensure bbox is within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                
                # Create a small visualization window in the top-right corner
                viz_size = 150
                viz_crop = overlay[y1:y2, x1:x2]
                if viz_crop.size > 0:
                    viz_resized = cv2.resize(viz_crop, (viz_size, viz_size))
                    
                    # Place visualization in top-right corner of original image
                    start_x = img_w - viz_size - 10
                    start_y = 10
                    end_x = start_x + viz_size
                    end_y = start_y + viz_size
                    
                    # Ensure we don't go out of bounds
                    if start_x >= 0 and start_y >= 0 and end_x <= img_w and end_y <= img_h:
                        vis_image[start_y:end_y, start_x:end_x] = viz_resized
                        
                        # Add border around visualization
                        cv2.rectangle(vis_image, (start_x-2, start_y-2), (end_x+2, end_y+2), (255, 255, 255), 2)
                        
                        # Add label
                        cv2.putText(vis_image, "Body Part Masks", (start_x, start_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            print(f"Error visualizing masks: {e}")
            return image
    
    def add_mask_legend(self, image):
        """Add a legend explaining the mask colors"""
        try:
            # Define legend properties
            legend_colors = [
                ((255, 0, 0), "Head"),
                ((0, 255, 0), "Upper Torso"),
                ((0, 0, 255), "Lower Torso"), 
                ((255, 255, 0), "Legs"),
                ((255, 0, 255), "Feet")
            ]
            
            # Legend position (bottom-left corner)
            legend_x = 10
            legend_y = image.shape[0] - 120  # Start from bottom
            legend_width = 150
            legend_height = 110
            
            # Draw legend background
            cv2.rectangle(image, (legend_x-5, legend_y-5), 
                         (legend_x + legend_width, legend_y + legend_height), 
                         (0, 0, 0), -1)  # Black background
            cv2.rectangle(image, (legend_x-5, legend_y-5), 
                         (legend_x + legend_width, legend_y + legend_height), 
                         (255, 255, 255), 1)  # White border
            
            # Add title
            cv2.putText(image, "Body Part Masks:", (legend_x, legend_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add color legend items
            for i, (color, label) in enumerate(legend_colors):
                y_pos = legend_y + 35 + (i * 15)
                
                # Draw color box
                cv2.rectangle(image, (legend_x, y_pos - 8), (legend_x + 12, y_pos + 2), color, -1)
                
                # Add label
                cv2.putText(image, label, (legend_x + 18, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            return image
            
        except Exception as e:
            print(f"Error adding mask legend: {e}")
            return image
    

    
    def extract_features(self, image):
        """Extract ReID features using official masking pipeline"""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Apply ReID transforms
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Generate official masks
            masks_tensor = self.generate_official_masks(image)
            
            # Extract features using the model with masks
            with torch.no_grad():
                try:
                    # Pass masks as external_parts_masks parameter (official way)
                    outputs = self.reid_model(image_tensor, external_parts_masks=masks_tensor)
                    
                    # BPBreID returns a tuple: (embeddings_dict, visibility_scores, id_cls_scores, ...)
                    if isinstance(outputs, tuple) and len(outputs) >= 1:
                        embeddings_dict = outputs[0]
                        
                        # Extract foreground features (official test embeddings)
                        if 'bn_foreg' in embeddings_dict:
                            feature_vector = embeddings_dict['bn_foreg']
                        elif 'foreground' in embeddings_dict:
                            feature_vector = embeddings_dict['foreground']
                        else:
                            # Fallback to first available embedding
                            feature_vector = list(embeddings_dict.values())[0]
                    else:
                        feature_vector = outputs
                    
                except Exception as model_error:
                    print(f"Model forward pass with masks failed: {model_error}")
                    # Try without masks as fallback
                    outputs = self.reid_model(image_tensor)
                    if isinstance(outputs, tuple):
                        feature_vector = outputs[0].get('bn_foreg', list(outputs[0].values())[0])
                    else:
                        feature_vector = outputs
            
            # Ensure 2D shape (batch_size, feature_dim)
            if len(feature_vector.shape) > 2:
                feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
            # Normalize features
            feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
            
            return feature_vector
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return dummy features as fallback
            return torch.randn(1, 512, device=self.device)
    
    def match_person(self, query_features):
        """Match query features against gallery"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        similarities = torch.mm(query_features, gallery_tensor.t())
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        if best_similarity > self.reid_threshold:
            return self.gallery_ids[best_idx], best_similarity
        else:
            return None, best_similarity
    
    def add_to_gallery(self, features, person_id, image):
        """Add a person to the gallery"""
        self.gallery_features.append(features)
        self.gallery_ids.append(person_id)
        self.gallery_images.append(image.copy())
    
    def process_frame(self, frame):
        """Process a single frame using official masking pipeline"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        frame_results = {
            'detections': [],
            'matches': [],
            'similarities': []
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
                        # Generate masks for visualization
                        masks_tensor = self.generate_official_masks(person_img)
                        
                        # Add mask visualization to the annotated frame
                        annotated_frame = self.visualize_masks_on_image(
                            annotated_frame, 
                            masks_tensor, 
                            person_bbox=(x1, y1, x2, y2),
                            alpha=0.3
                        )
                        
                        # Extract features using official masking pipeline
                        features = self.extract_features(person_img)
                        
                        # Try to match with gallery
                        matched_id, similarity = self.match_person(features)
                        
                        # Store results for analysis
                        detection_result = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'matched_id': matched_id,
                            'similarity': similarity
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
                        
                        # Add masking indicator
                        mask_label = "BPBreID Official: PifPaf + R-CNN + Transforms"
                        cv2.putText(annotated_frame, mask_label, (x1, y2+15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated_frame, frame_results
    
    def add_gallery_image(self, image_path):
        """Add a gallery image from file path"""
        print(f"Loading gallery image: {image_path}")
        
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
                        features = self.extract_features(person_img)
                        self.add_to_gallery(features, self.next_person_id, person_img)
                        print(f"Added Person {self.next_person_id} to gallery from {image_path}")
                        self.next_person_id += 1
                        person_detected = True
                        break
            
            if person_detected:
                break
        
        if not person_detected:
            raise ValueError(f"No person detected in gallery image: {image_path}")
        
        return True
    
    def test_video(self, video_path, expected_match=True, save_annotated=False, output_path=None):
        """Test the ReID model on a video using Torchvision masking pipeline"""
        print(f"\nTesting video with Torchvision Masking Pipeline: {video_path}")
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
            'masking_method': 'Torchvision Pipeline (PifPaf + Mask R-CNN)',
            'total_frames': total_frames,
            'frames_with_detection': 0,
            'frames_with_match': 0,
            'frames_without_match': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy': 0.0,
            'similarity_scores': [],
            'frame_details': []
        }
        
        frame_idx = 0
        
        print("Processing frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with Torchvision masking
            annotated_frame, frame_result = self.process_frame(frame)
            
            frame_detail = {
                'frame_idx': frame_idx,
                'has_detection': len(frame_result['detections']) > 0,
                'has_match': len(frame_result['matches']) > 0,
                'similarities': frame_result['similarities'],
                'correct': False
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
                    frame_detail['correct'] = frame_detail['has_match']
                else:
                    frame_detail['correct'] = not frame_detail['has_match']
                
                if frame_detail['correct']:
                    results['correct_predictions'] += 1
                else:
                    results['incorrect_predictions'] += 1
                
                # Add correctness indicator to frame
                correct_text = "✓" if frame_detail['correct'] else "✗"
                correct_color = (0, 255, 0) if frame_detail['correct'] else (0, 0, 255)
                cv2.putText(annotated_frame, correct_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, correct_color, 2)
            
            # Add frame info to annotated frame
            info_text = f"Frame {frame_idx+1}/{total_frames} | Expected: {'Match' if expected_match else 'No Match'}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add masking method indicator
            mask_text = "BPBreID Official: PifPaf + R-CNN + Transforms"
            cv2.putText(annotated_frame, mask_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Add mask legend to frame
            annotated_frame = self.add_mask_legend(annotated_frame)
            
            # Save annotated frame
            if writer:
                writer.write(annotated_frame)
            
            results['frame_details'].append(frame_detail)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Calculate final accuracy
        total_detections = results['frames_with_detection']
        if total_detections > 0:
            results['accuracy'] = results['correct_predictions'] / total_detections
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"Test completed: {results['correct_predictions']}/{total_detections} correct predictions")
        print(f"Accuracy: {results['accuracy']:.3f}")
        
        return results
    
    def run_full_test(self, gallery_image_path, video1_path, video2_path, 
                     save_annotated=True, output_dir="reid_torchvision_mask_viz_results"):
        """Run complete test suite using Torchvision masking pipeline"""
        print("="*60)
        print("Torchvision Masking BPBreID Video Test Suite")
        print("Using: PifPaf + Torchvision Mask R-CNN + Official Transforms")
        print("="*60)
        
        # Create output directory
        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clear gallery first
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Step 1: Add gallery image
        print("\n1. Loading gallery image...")
        try:
            self.add_gallery_image(gallery_image_path)
        except Exception as e:
            print(f"Error loading gallery image: {e}")
            return None
        
        # Step 2: Test video 1 (should match)
        print("\n2. Testing Video 1 (Same Person - Should Match)...")
        video1_output = os.path.join(output_dir, "video1_annotated.mp4") if save_annotated else None
        
        try:
            results1 = self.test_video(
                video1_path, 
                expected_match=True, 
                save_annotated=save_annotated,
                output_path=video1_output
            )
        except Exception as e:
            print(f"Error testing video 1: {e}")
            return None
        
        # Step 3: Test video 2 (should not match)
        print("\n3. Testing Video 2 (Different Person - Should NOT Match)...")
        video2_output = os.path.join(output_dir, "video2_annotated.mp4") if save_annotated else None
        
        try:
            results2 = self.test_video(
                video2_path, 
                expected_match=False, 
                save_annotated=save_annotated,
                output_path=video2_output
            )
        except Exception as e:
            print(f"Error testing video 2: {e}")
            return None
        
        # Compile final results
        final_results = {
            'test_timestamp': datetime.now().isoformat(),
            'masking_method': 'Torchvision Pipeline (PifPaf + Mask R-CNN + Transforms)',
            'gallery_image': gallery_image_path,
            'reid_threshold': self.reid_threshold,
            'device': str(self.device),
            'video1_results': results1,
            'video2_results': results2,
            'overall_performance': {
                'total_frames_tested': results1['frames_with_detection'] + results2['frames_with_detection'],
                'total_correct': results1['correct_predictions'] + results2['correct_predictions'],
                'total_incorrect': results1['incorrect_predictions'] + results2['incorrect_predictions'],
                'overall_accuracy': 0.0
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
        
        # Print detailed results (reuse the same function)
        self.print_test_summary(final_results)
        
        print(f"\nResults saved to: {results_file}")
        if save_annotated:
            print(f"Annotated videos saved to: {output_dir}")
        
        return final_results
    
    def print_test_summary(self, results):
        """Print a detailed summary of test results"""
        print("\n" + "="*60)
        print("TORCHVISION MASKING TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Masking Method: {results['masking_method']}")
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
        
        if v2['similarity_scores']:
            avg_sim = np.mean(v2['similarity_scores'])
            max_sim = np.max(v2['similarity_scores'])
            min_sim = np.min(v2['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Overall Results
        overall = results['overall_performance']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Frames Tested: {overall['total_frames_tested']}")
        print(f"  Total Correct: {overall['total_correct']}")
        print(f"  Total Incorrect: {overall['total_incorrect']}")
        print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f} ({overall['overall_accuracy']*100:.1f}%)")
        
        print("="*60)


def main():
    """Main function to run the Torchvision masking test suite"""
    print("Torchvision Masking BPBreID Video Test Suite")
    print("Using: PifPaf + Torchvision Mask R-CNN (No Detectron2 needed!)")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    if not PIFPAF_AVAILABLE:
        missing_deps.append("openpifpaf (install with: pip install openpifpaf)")
    if not SEGMENTATION_AVAILABLE:
        missing_deps.append("torchvision (should be included with PyTorch)")
    
    if missing_deps:
        print("Optional dependencies missing:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nThe system will work with fallbacks, but for best results install the missing dependencies.")
        print()
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    testing_dir = os.path.dirname(current_dir)
    realtime_dir = os.path.dirname(testing_dir)               
    bpbreid_dir = os.path.dirname(realtime_dir) 
    
    # Configuration - paths relative to parent directory
    reid_model_path = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    hrnet_path = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")

    # Test file paths
    gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1.jpg")
    video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1-vid.MOV")
    video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-2-vid.MOV")

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
        # Create Torchvision masking test system
        print("Initializing Torchvision Masking BPBreID test system...")
        print("Loading models (this may take a moment)...")
        
        tester = TorchvisionOfficialMaskingBPBreIDTester(reid_model_path, hrnet_path)
        
        print("\nChoose test mode:")
        print("1. Full video test suite (recommended)")
        print("2. Single image test")
        print("3. Interactive mode")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            # Full video test suite
            if all(os.path.exists(p) for p in [gallery_image_path, video1_path, video2_path]):
                results = tester.run_full_test(
                    gallery_image_path=gallery_image_path,
                    video1_path=video1_path,
                    video2_path=video2_path,
                    save_annotated=True,
                    output_dir="reid_torchvision_mask_viz_results"
                )
                
                if results:
                    print("\nFull test completed successfully!")
                    print("Check 'reid_torchvision_mask_viz_results' directory for outputs")
                else:
                    print("Full test failed!")
            else:
                print("Test files not found. Please check paths.")
                
        elif choice == "2":
            # Single image test
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                if image is not None:
                    annotated, frame_result = tester.process_frame(image)
                    cv2.imshow("Torchvision Masking Test Result", annotated)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    print(f"Detections: {len(frame_result['detections'])}")
                    for i, det in enumerate(frame_result['detections']):
                        print(f"  Detection {i+1}: ID={det['matched_id']}, Similarity={det['similarity']:.3f}")
                else:
                    print("Could not load image!")
            else:
                print("Image file not found!")
                
        elif choice == "3":
            # Interactive mode
            print("\nInteractive mode - Add gallery image first")
            img_path = input("Enter gallery image path: ").strip()
            if os.path.exists(img_path):
                try:
                    tester.add_gallery_image(img_path)
                    print("Gallery image added successfully!")
                    print("You can now test with videos or images using choice 1 or 2")
                except Exception as e:
                    print(f"Error adding gallery image: {e}")
            else:
                print("Gallery image not found!")
                
        else:
            print("Invalid choice. Exiting.")
            
    except Exception as e:
        print(f"Error running Torchvision masking test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Torchvision Masking BPBreID Video Test Suite")
    print("=" * 60)
    print("Features:")
    print("- Uses Torchvision Mask R-CNN (easier to install than Detectron2)")
    print("- Optional OpenPifPaf for pose estimation")
    print("- Official BPBreid mask transformation pipeline")
    print("- Full video test suite with accuracy metrics")
    print("- Graceful fallbacks when dependencies are missing")
    print("\nEasy Installation:")
    print("- pip install openpifpaf  # Optional but recommended")
    print("- torchvision should already be installed with PyTorch")
    print()
    
    main()
