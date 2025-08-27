#!/usr/bin/env python3
"""
Test script to verify BPBreID model loading on Jetson device
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

def test_model_loading():
    """Test if the BPBreID model can be loaded correctly"""
    
    print("Testing BPBreID model loading...")
    
    try:
        # Import the necessary components
        from bpbreid_yolo_masked_reid_fin2 import ImprovedBPBreIDYOLOMaskedReID
        
        # Check if model files exist
        reid_model_path = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
        hrnet_path = "pretrained_models/hrnetv2_w32_imagenet_pretrained.pth"
        
        if not os.path.exists(reid_model_path):
            print(f"Error: BPBreID model not found at {reid_model_path}")
            return False
            
        if not os.path.exists(hrnet_path):
            print(f"Error: HRNet model not found at {hrnet_path}")
            return False
        
        print("Model files found, attempting to load...")
        
        # Initialize the re-identifier
        reid_system = ImprovedBPBreIDYOLOMaskedReID(
            reid_model_path=reid_model_path,
            hrnet_path=hrnet_path,
            yolo_model_path='yolov8n-pose.pt'
        )
        
        print("✅ BPBreID model loaded successfully!")
        print(f"Device: {reid_system.device}")
        print(f"Model type: {type(reid_system.model)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading BPBreID model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test passed!")
    else:
        print("\n💥 Model loading test failed!")
        sys.exit(1)
