# Custom YOLOv8 Vehicle Detection

A Google Colab notebook for training a YOLOv8 object detection model on a custom vehicle dataset. The notebook handles the full pipeline — from loading data to training, validating, and exporting the final model weights.

**Detects:** `bicycle`, `car`, `bike`  
**Base Model:** YOLOv8 Medium (`yolov8m.pt`)  
**Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## What This Notebook Does

1. Verifies GPU availability on Colab
2. Mounts Google Drive and loads the dataset config
3. Clears stale cache files from previous runs
4. Trains YOLOv8 on the custom dataset (`100 epochs, imgsz=640`)
5. Displays training results — confusion matrix and metric curves
6. Runs inference on the validation set and previews predictions
7. Backs up weights and prediction outputs to Google Drive
8. Packages the final model into a downloadable `.zip`

---

## Dataset Structure

Your dataset must follow this layout in Google Drive:

```
MyProject/
└── DataSet/
    ├── data.yaml
    ├── classes.txt
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

**`data.yaml`**
```yaml
path: /content/drive/MyDrive/MyProject/DataSet
train: train/images
val:   val/images
nc: 3
names: ['bicycle', 'car', 'bike']
```

**`classes.txt`**
```
bicycle
car
bike
```

Labels must be in YOLO format: `class_id cx cy w h` (normalized 0–1), one object per line.

---

## How to Reuse for Your Own Dataset

Replace the following paths and values in the notebook:

### 1. Dataset config paths
```python
path_to_classes_txt = '/content/drive/MyDrive/YOUR_PROJECT/DataSet/classes.txt'
path_to_data_yaml   = '/content/drive/MyDrive/YOUR_PROJECT/DataSet/data.yaml'
```

### 2. Cache deletion paths
```bash
!rm -f /content/drive/MyDrive/YOUR_PROJECT/DataSet/train/images.cache
!rm -f /content/drive/MyDrive/YOUR_PROJECT/DataSet/val/images.cache
!rm -f /content/drive/MyDrive/YOUR_PROJECT/DataSet/train/labels.cache
!rm -f /content/drive/MyDrive/YOUR_PROJECT/DataSet/val/labels.cache
```

### 3. Training command
```bash
!yolo detect train \
  data=/content/drive/MyDrive/YOUR_PROJECT/DataSet/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  patience=0 \
  project=/content/drive/MyDrive/YOUR_PROJECT/runs
```
Adjust `model`, `epochs`, and `imgsz` as needed for your task.

### 4. Inference path
```bash
!yolo detect predict \
  model=/content/drive/MyDrive/YOUR_PROJECT/runs/train/weights/best.pt \
  source=/content/drive/MyDrive/YOUR_PROJECT/DataSet/val/images \
  save=True
```

### 5. Backup paths (weights + predictions)
```python
src = "/content/drive/MyDrive/YOUR_PROJECT/runs/train/weights/"
dst = "/content/drive/MyDrive/YOUR_PROJECT/runs/trained_model/weights/"
```
```python
src_dir = "/content/runs/detect/predict"
dst_dir = "/content/drive/MyDrive/YOUR_PROJECT/runs/validated/outputs/predict"
```

Also update `data.yaml` with your actual class names and count (`nc`).

---

## Requirements

- Google Colab with GPU enabled (`Runtime → Change runtime type → GPU`)
- Dataset stored in Google Drive in the structure shown above
- Labels annotated in YOLO format

```bash
pip install ultralytics
```

---

## Output

After training, the following are saved to your Drive:

```
YOUR_PROJECT/runs/train/
├── weights/
│   ├── best.pt   ← use this for inference
│   └── last.pt
├── results.png
└── confusion_matrix_normalized.png
```

---

## Notes

- For training, you should prefer `Custom_YOLOv8(lst).ipynb` file as this file is best among all other for training purpose. Just change with your parameter. 
- `patience=0` disables early stopping. Set it to a positive value (e.g., `20`) if you want training to stop when validation metrics plateau.
- `best.pt` is the recommended weight file for deployment or further fine-tuning.
- All outputs are saved directly to Google Drive so progress is not lost if the Colab session disconnects.

## Testing or Inferencing your trained model
- You can use the code inside Test folder with the file name `inference.ipynb`.
- It is might not show in github workspace but you can use it by downloading the file.
- Replace the path of input with your input path.
- You can test your model on image, video and live with your webcam.
