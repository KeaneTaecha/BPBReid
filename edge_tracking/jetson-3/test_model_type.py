#!/usr/bin/env python3
"""
Test script to verify what type of model is being created
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_model_creation():
    print("Testing model creation...")
    
    try:
        # Import torchreid
        import torchreid
        
        print("Available models:")
        torchreid.models.show_avai_models()
        
        # Try to create BPBreID model
        print("\nTrying to create BPBreID model...")
        
        # Create a simple config
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = '../../pretrained_models/'
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
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
        
        # Mask configuration
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.softmax_weight = 1.0
        config.model.bpbreid.masks.background_computation_strategy = 'threshold'
        config.model.bpbreid.masks.mask_filtering_threshold = 0.3
        
        # Try building model
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=751,
            config=config,
            pretrained=True
        )
        
        print(f"✅ Model created successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Model class: {model.__class__.__name__}")
        print(f"Model module: {model.__class__.__module__}")
        
        # Check if it's actually BPBreID
        if 'BPBreID' in str(type(model)):
            print("✅ Model is BPBreID!")
        else:
            print("❌ Model is NOT BPBreID!")
            print(f"Model string representation: {str(type(model))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation()
    if success:
        print("\n🎉 Model creation test completed!")
    else:
        print("\n💥 Model creation test failed!")
        sys.exit(1)
