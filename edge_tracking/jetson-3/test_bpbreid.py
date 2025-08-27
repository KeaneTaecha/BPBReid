#!/usr/bin/env python3
"""
Test script to verify BPBreID initialization and gallery loading
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from person_tracker import BPBreIDReidentifier

def test_bpbreid():
    print("Testing BPBreID initialization...")
    
    try:
        # Initialize BPBreID
        bpbreid = BPBreIDReidentifier()
        
        if bpbreid.reidentifier is None:
            print("❌ BPBreID initialization failed")
            return False
        
        print("✅ BPBreID initialization successful")
        
        # Test gallery loading
        print("\nTesting gallery loading...")
        bpbreid.load_gallery_persons("gallery_folder")
        
        print(f"✅ Gallery loading completed. Loaded {len(bpbreid.gallery_features)} persons")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bpbreid()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
