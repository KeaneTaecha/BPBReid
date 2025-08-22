# BPBreID Official Mask ReID Script

This script implements the official BPBreID testing pipeline with proper mask usage for real-time person re-identification.

## Overview

The script `bpbreid_official_mask_reid_simple.py` applies the official BPBreID masking pipeline to:

1. **Extract features from a gallery image** (e.g., `person-1.jpg`)
2. **Process video frames** with official mask creation using MaskRCNN
3. **Compare features** between gallery image and video frames
4. **Show real-time ReID results** with similarity scores

## Key Features

- **Official mask filtering** during testing (`mask_filtering_testing = True`)
- **Proper external mask usage** (`test_use_target_segmentation = 'soft'`)
- **Official mask transforms** using `'five_v'` preprocessing
- **Correct feature extraction** and similarity computation
- **Real-time video processing** with YOLO detection
- **MaskRCNN-based mask creation** following the official pipeline

## Requirements

The script requires the following files to be present:

- `pretrained_models/bpbreid_market1501_hrnet32_10642.pth` - BPBreID model weights
- `pretrained_models/hrnet32_imagenet.pth` - HRNet backbone weights
- `yolov8n.pt` - YOLO model for person detection
- `datasets/Compare/dataset-2/person-1.jpg` - Gallery image
- `datasets/Compare/dataset-2/person-1-vid.MOV` - Video to process

## Usage

```bash
cd realtime
python bpbreid_official_mask_reid_simple.py
```

## How It Works

### 1. Model Initialization

The script loads:
- **BPBreID model** with official testing configuration
- **MaskRCNN model** for official mask creation
- **YOLO model** for person detection

### 2. Gallery Feature Extraction

1. Loads the gallery image (`person-1.jpg`)
2. Creates a full mask for the gallery image
3. Extracts both foreground and parts features using BPBreID

### 3. Video Processing

For each video frame:
1. **Detect persons** using YOLO
2. **Create masks** using the official MaskRCNN method
3. **Extract features** from each detected person
4. **Compute similarities** with gallery features
5. **Visualize results** with color-coded bounding boxes

### 4. Similarity Computation

The script computes:
- **Foreground similarity** (60% weight)
- **Parts similarity** (40% weight) - average across 5 body parts
- **Combined similarity** - weighted average of both

### 5. Visualization

Results are color-coded:
- **Green**: High similarity (>0.7)
- **Yellow**: Medium similarity (0.5-0.7)
- **Red**: Low similarity (<0.5)

## Configuration

The script uses the official BPBreID testing configuration:

```python
config.model.bpbreid.mask_filtering_testing = True
config.model.bpbreid.test_use_target_segmentation = 'soft'
config.model.bpbreid.masks.preprocess = 'five_v'
config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']
```

## Output

The script generates:
- **Real-time preview** showing detection and similarity results
- **Output video** with annotated results (if `save_video=True`)
- **Console output** with processing statistics

## File Structure

```
realtime/
├── bpbreid_official_mask_reid_simple.py  # Main script
├── README_BPBreID_Official_Mask_ReID.md  # This file
└── yolov8n.pt                            # YOLO model
```

## Troubleshooting

### Common Issues

1. **Model files not found**: Ensure all required model files are in the correct paths
2. **MaskRCNN loading fails**: The script will fall back to YOLO bounding box masks
3. **CUDA out of memory**: Reduce batch size or use CPU processing
4. **Video file not found**: Check the video path in the script

### Performance Tips

- Use GPU for faster processing
- Reduce video resolution if processing is slow
- Adjust YOLO confidence threshold for fewer detections
- Use `show_preview=False` for faster processing without visualization

## Technical Details

### Mask Processing

The script follows the official BPBreID mask processing pipeline:

1. **MaskRCNN detection** for person segmentation
2. **5-part vertical division** (`five_v` preprocessing)
3. **Background mask addition** with threshold-based computation
4. **Soft masking** during testing

### Feature Extraction

Features are extracted using:
- **Foreground features**: Global person representation
- **Parts features**: 5 body part representations
- **Combined similarity**: Weighted average of both

### Model Architecture

The script uses the official BPBreID architecture:
- **Backbone**: HRNet-32
- **Pooling**: GWAP (Global Weighted Average Pooling)
- **Normalization**: Identity
- **Dimension reduction**: After pooling (512D)

## Comparison with Other Methods

This script differs from other implementations by:
- Using the **official mask creation pipeline** from `get_labels.py`
- Following the **exact testing configuration** from the official BPBreID paper
- Implementing **proper mask filtering** during testing
- Using **soft masking** instead of hard masking
- Computing **both foreground and parts similarities**

## References

- BPBreID: Body Part-based Re-identification for Person Re-identification
- Official BPBreID implementation in `torchreid/`
- MaskRCNN for person segmentation
- YOLO for person detection
