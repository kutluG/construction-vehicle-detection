# 🏗️ Construction Site Object Detection

A YOLO11-based object detection system for construction sites, trained to detect machinery, vehicles, people, and safety equipment violations.

## 🎯 Features

- **Multi-class Detection**: Detects 10 different classes
  - Construction machinery
  - Vehicles
  - People/Workers
  - Safety equipment (Hardhat, Mask, Safety Vest)
  - Safety violations (NO-Hardhat, NO-Mask, NO-Safety Vest)
  - Safety cones

- **Web Interface**: User-friendly Streamlit app with drag-and-drop functionality
- **High Accuracy**: 
  - mAP50-95: 56.7%
  - mAP50: 84.5%
  - Trained on 2,603 construction site images

## 📋 Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended) or CPU
- 8GB+ RAM

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/kutluG/construction-vehicle-detection.git
cd construction-vehicle-detection
```

### 2. Automated Setup (Windows)

**Easy way** - Run the setup script which automatically installs dependencies and downloads the model:
```bash
setup.bat
```

The setup script will:
- ✓ Check Python installation
- ✓ Install required packages
- ✓ Automatically download the trained model (109 MB) from GitHub releases
- ✓ Create necessary folder structure

### 3. Manual Setup (All Platforms)

**Step 1: Install Dependencies**

**Option A: Using pip (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual installation**
```bash
pip install torch ultralytics opencv-python streamlit pillow numpy
```

**Step 2: Download the Trained Model**

**Important**: The trained model weights are not included in the repository due to file size.

**Download the pre-trained YOLO11x model:**

1. Automatic download (Windows):
   - Already done if you ran `setup.bat`
   
2. Manual download:
   ```bash
   # Windows PowerShell
   Invoke-WebRequest -Uri "https://github.com/kutluG/construction-vehicle-detection/releases/download/v1.0.0/best.pt" -OutFile "runs/detect/y11x_10ep/weights/best.pt"
   
   # Linux/Mac
   mkdir -p runs/detect/y11x_10ep/weights
   wget https://github.com/kutluG/construction-vehicle-detection/releases/download/v1.0.0/best.pt -P runs/detect/y11x_10ep/weights/
   ```

3. Or download from browser:
   - Go to [Releases](https://github.com/kutluG/construction-vehicle-detection/releases/tag/v1.0.0)
   - Download `best.pt` (109 MB)
   - Create folders: `runs/detect/y11x_10ep/weights/`
   - Place the file there

**Alternative: Train your own model** (requires dataset)
```bash
python train.py --weights yolo11x.pt --data data_vehicles.yaml --epochs 10 --batch 4
```

### 4. Run the Web Application

**Windows:**
```bash
# Double-click run_app.bat
# OR run in terminal:
streamlit run app.py
```

**Mac/Linux:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 5. Using Command Line (Alternative)

Process a single image:
```bash
python main.py path/to/image.jpg --conf 0.25 --imgsz 1920
```

Process all images in a folder:
```bash
python main.py input/*.jpg
```

## 📁 Project Structure

```
davoski-yolo/
├── app.py                          # Streamlit web interface
├── main.py                         # Command-line detection script
├── train.py                        # Training script
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows launcher
├── data_vehicles.yaml              # Dataset configuration
├── input/                          # Input images folder
├── output/                         # Processed images output
├── runs/detect/y11x_10ep/          # Trained model location
│   └── weights/
│       └── best.pt                 # YOLO11x trained weights (download from releases)
└── dataset/                        # Training dataset (optional)
```

## 🎮 Usage

### Web Interface

1. Open the app (http://localhost:8501)
2. Drag & drop an image or click "Browse files"
3. Adjust settings:
   - **Confidence Threshold**: 0.15-0.6 (default: 0.25)
   - **Image Size**: 640/1280/1920 (default: 1920)
4. Click "🔍 Process Image"
5. View results and download the annotated image

### Command Line

```bash
# Basic usage
python main.py image.jpg

# Custom confidence threshold
python main.py image.jpg --conf 0.3

# Higher resolution processing
python main.py image.jpg --imgsz 1920

# Use different model
python main.py image.jpg --model path/to/model.pt
```

## 🔧 Troubleshooting

### Model Not Found Error
- Download the trained model weights from [GitHub Releases](https://github.com/kutluG/construction-vehicle-detection/releases/tag/v1.0.0)
- Verify the file exists at: `runs/detect/y11x_10ep/weights/best.pt`

### CUDA/GPU Issues
- The model will automatically use CPU if GPU is not available
- For GPU training, install CUDA-compatible PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### Import Errors
- Make sure all requirements are installed: `pip install -r requirements.txt`
- Try upgrading pip: `python -m pip install --upgrade pip`

## 📊 Model Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Overall | 86.2% | 66.1% | 73.8% | 38.2% |
| Hardhat | 92.6% | 74.7% | 82.3% | 47.9% |
| Machinery | 75.7% | 76.4% | 82.5% | 46.1% |
| Mask | 94.3% | 85.7% | 86.0% | 49.9% |

Trained for 10 epochs (33 minutes) on NVIDIA RTX 4060.

## 🎓 Training Your Own Model

If you want to train with your own dataset:

1. Prepare dataset in YOLO format (images + labels)
2. Update `data_vehicles.yaml` with your paths
3. Run training:
   ```bash
   python train.py --weights yolo11s.pt --data your_data.yaml --epochs 100
   ```

## 📝 Notes

- **First run**: Model loading takes 5-10 seconds
- **Processing time**: ~0.1-0.2 seconds per image (GPU) or 1-3 seconds (CPU)
- **Best results**: Use well-lit, clear construction site images
- **Confidence tuning**: Lower threshold (0.15-0.25) for more detections, higher (0.4-0.6) for precision

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines if needed]

## 📧 Contact

[Add your contact information]
