#!/usr/bin/env python3
"""
Check torchreid version and available models
"""

import torch
import torchreid
import sys

def check_torchreid():
    """Check torchreid installation and available models"""
    
    print("=== TorchReID Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchReID version: {torchreid.__version__ if hasattr(torchreid, '__version__') else 'Unknown'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    print("\n=== Available Models ===")
    try:
        from torchreid import models
        models.show_avai_models()
    except Exception as e:
        print(f"Error showing available models: {e}")
    
    print("\n=== BPBreID Model Check ===")
    try:
        from torchreid import models
        if 'bpbreid' in dir(models):
            print("✅ BPBreID model is available")
            print(f"BPBreID function: {models.bpbreid}")
        else:
            print("❌ BPBreID model is not available")
    except Exception as e:
        print(f"Error checking BPBreID model: {e}")
    
    print("\n=== Build Model Function Check ===")
    try:
        from torchreid import models
        import inspect
        sig = inspect.signature(models.build_model)
        print(f"build_model signature: {sig}")
        print(f"build_model docstring: {models.build_model.__doc__}")
    except Exception as e:
        print(f"Error checking build_model function: {e}")

if __name__ == "__main__":
    check_torchreid()
