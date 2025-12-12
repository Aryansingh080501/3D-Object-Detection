# 3D-Object-Detection-across-KITTI-nuScenes-and-Custom-PCD-Dataset

This project implements a full 3D object detection inference pipeline using multiple pretrained models across multiple datasets.  
The work includes custom dataset integration, visualization tools, model comparison, and export of all inference artifacts and video demos.

All modifications, dataset extensions, and pipeline engineering were implemented by me.

---

## **1. Project Features**

- Inference using **four** pretrained models:
  - PointPillars (KITTI)
  - 3DSSD (KITTI)
  - PointPillars (nuScenes)
  - CenterPoint (nuScenes)
- Evaluation across **three** datasets:
  - KITTI
  - nuScenes
  - Custom PCD dataset (lamppost, roomscan1, roomscan2)
- Custom dataset support implemented via:
  - `.pcd` loading using Open3D  
  - Normalization, scaling, and downsampling  
  - Conversion to KITTI-style `.bin` format  
  - Integration into MMDetection3D voxelization pipeline
- Export of:
  - `.ply` point clouds
  - `.ply` predicted boxes
  - `.json` prediction metadata
  - `.png` rendered outputs
- Open3D visualization enhancements:
  - Added `--camera-view {iso,front,top,side}`  
- Demo video generation using MoviePy  
- Clean, reproducible evaluation workflow

---

## **2. Environment Setup**

### **Requirements**
- Python 3.10
- NVIDIA GPU with CUDA (GTX 1050 used here)
- pip, virtual environment recommended

### **Installation Commands**
```bash
python -m pip install -U pip
pip install openmim open3d moviepy tqdm matplotlib seaborn pandas
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
mim install mmengine
pip install mmcv==2.1.0 mmdet==3.2.0
mim install mmdet3d
```

## 3. Running Inference

Below are the commands used to run inference for each model and dataset.  
All artifacts (.png, .ply, .json) are automatically saved to the specified `--out-dir`.

---

### **KITTI – PointPillars**
```bash
python mmdet3d_inference2.py \
  --dataset kitti \
  --input-path data/kitti/training \
  --frame-number 000008 \
  --model checkpoints/kitti_pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  --checkpoint checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.pth \
  --out-dir outputs/kitti_pointpillars_gpu \
  --device cuda:0 \
  --score-thr 0.2 \
  --headless
```
### **KITTI – 3DSSD**
```bash
python mmdet3d_inference2.py \
  --dataset kitti \
  --input-path data/kitti/training \
  --frame-number 000008 \
  --model checkpoints/3dssd/3dssd_4x4_kitti-3d-car.py \
  --checkpoint checkpoints/3dssd/3dssd_4x4_kitti-3d-car.pth \
  --out-dir outputs/3dssd_gpu \
  --device cuda:0 \
  --score-thr 0.6 \
  --headless
```
### **nuScenes – PointPillars**
```bash
python mmdet3d_inference2.py \
  --dataset any \
  --input-path data/nuscenes_demo/lidar/sample.pcd.bin \
  --model checkpoints/nuscenes_pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
  --checkpoint checkpoints/nuscenes_pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_nus-3d.pth \
  --out-dir outputs/nuscenes_pointpillars_gpu \
  --device cuda:0 \
  --score-thr 0.2 \
  --headless
```
### **Custom PCD Dataset**
```bash
python mmdet3d_inference2.py \
  --dataset custom_pcd \
  --input-path data/custom_pcd \
  --model checkpoints/kitti_pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  --checkpoint checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.pth \
  --out-dir outputs/custom_pcd_kitti_pointpillars_gpu \
  --device cuda:0 \
  --score-thr 0.1 \
  --headless \
  --max-samples 3
```

## 4. Visualization

Inference outputs include:
- 3D point clouds (`*_points.ply`)
- Predicted bounding boxes (`*_pred_bboxes.ply`)
- JSON prediction files (`*_predictions.json`)
- 2D overlays (for KITTI/nuScenes)
- Coordinate axes (`*_axes.ply`)

To generate 3D visualization screenshots, use the enhanced Open3D viewer script:

### **Open3D Screenshot Command**
```bash
python scripts/open3d_view_saved_ply.py \
  --dir outputs/custom_pcd_kitti_pointpillars_gpu \
  --basename lamppost \
  --width 1600 --height 1200 \
  --camera-view iso \
  --save-path results/custom_pcd_lamppost_iso.png \
  --no-show
```

## 5. Results Summary

### Detection Performance Table

| Model            | Dataset      | Detection Count | Mean Score | High-Conf (≥0.7) | Notes |
|------------------|--------------|-----------------|------------|------------------|-------|
| PointPillars     | KITTI        | Low             | **0.79**   | Strong           | Best KITTI performer |
| 3DSSD            | KITTI        | Many            | 0.25       | Weak             | High false positives |
| PointPillars     | nuScenes     | Highest count   | 0.33       | Low              | Detects many small objects |
| CenterPoint      | nuScenes     | Moderate        | 0.41       | **Highest**      | Best on nuScenes |
| PointPillars     | Custom PCD   | 0–2             | Very Low   | 0                | Poor generalization |
| CenterPoint      | Custom PCD   | Few             | Low        | 0                | Occasional hallucinations |

---

### Key Observations

- **KITTI-trained models** perform very well on structured road scenes but degrade sharply on non-driving datasets.
- **nuScenes-trained CenterPoint** provides the most consistent performance, even though confidence varies across scenes.
- On **Custom PCD**, all models struggle:
  - Little to no detections
  - Low confidence
  - Incorrect or hallucinated predictions
- The custom dataset reveals strong **domain dependence** and poor generalization beyond autonomous-driving LiDAR distributions.

## 6. Deliverables Overview

The final submission includes the following components:

### **1. REPORT.md**
A polished 1–2 page document containing:
- Environment setup  
- Commands used for inference  
- Models and datasets  
- Metrics table  
- Screenshots  
- Key takeaways and limitations  

### **2. results/**  
Contains:
- `demo_video.mp4` (stitched detection video)
- ≥4 labeled screenshots from:
  - KITTI
  - nuScenes
  - Custom PCD dataset
- Additional Open3D screenshots for qualitative comparison

### **3. Modified Code (Fully Commented)**
- `mmdet3d_inference2.py`
  - Added `custom_pcd` dataset mode
  - Implemented `.pcd` parsing, normalization, and KITTI `.bin` conversion
  - Added batch inference, downsampling, and error handling
- `scripts/open3d_view_saved_ply.py`
  - Added `--camera-view` argument and camera presets

### **4. README.md**
Contains:
- Installation instructions  
- Dataset preparation  
- Commands for all models/datasets  
- Visualization and video creation steps  
- Directory structure and reproducibility notes  

All outputs (.ply, .json, .png) are included under `outputs/`.


