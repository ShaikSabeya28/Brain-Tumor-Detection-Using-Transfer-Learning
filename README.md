# 🧠 Diagnosis of Brain Tumors Using Deep Segmentation and Transfer Learning

A deep learning-based medical image analysis project designed to assist in the **classification and detection of brain tumors from MRI images**.

The project combines **image segmentation using U-Net** with **transfer learning models**, including **EfficientNetB3, ResNet50, and InceptionV3**, to analyze MRI brain images and classify tumor categories.

---

## 🎯 Project Objective

Brain tumors require early and accurate diagnosis to support effective medical treatment.

The objective of this project is to develop a deep learning-based system that can:

* 🧠 Analyze MRI brain images
* 🔍 Segment relevant tumor regions
* 🤖 Extract meaningful image features
* 📊 Classify brain MRI images
* 📈 Compare multiple transfer learning architectures
* ⚕️ Assist medical image analysis

> **Note:** This project is intended for educational and research purposes and is not a replacement for professional medical diagnosis.

---

## 🌟 Features

* ✅ **MRI Image Processing**

  * Processes brain MRI images for deep learning analysis.

* 🔍 **Tumor Segmentation**

  * Uses **U-Net** to identify and segment relevant regions in MRI images.

* 🧠 **Deep Learning Classification**

  * Uses multiple transfer learning architectures.

* ⚡ **EfficientNetB3**

  * Used for efficient feature extraction and classification.

* 🔄 **ResNet50**

  * Uses residual learning to extract deep image features.

* 🏗️ **InceptionV3**

  * Uses multi-scale convolutional feature extraction.

* 📊 **Model Comparison**

  * Compares the performance of different deep learning models.

* 📈 **Performance Evaluation**

  * Uses classification metrics to evaluate model performance.

---

## 🛠️ Tech Stack

| Technology       | Role                           |
| ---------------- | ------------------------------ |
| **Python**       | Programming language           |
| **TensorFlow**   | Deep learning framework        |
| **Keras**        | Neural network development     |
| **OpenCV**       | Image processing               |
| **PIL**          | Image manipulation             |
| **NumPy**        | Numerical computation          |
| **Pandas**       | Data processing                |
| **Scikit-learn** | Model evaluation               |
| **Matplotlib**   | Visualization                  |
| **Seaborn**      | Data visualization             |
| **Google Colab** | Model development and training |
| **Google Drive** | Dataset storage                |

---

## 🧠 Deep Learning Architecture

The project uses two major stages:

```text
MRI Brain Images
       │
       ▼
Image Preprocessing
       │
       ▼
U-Net Segmentation
       │
       ▼
Tumor/Relevant Region
       │
       ▼
Feature Extraction
       │
       ├───────────────┐
       ▼               ▼
EfficientNetB3     ResNet50
       │               │
       └───────┬───────┘
               │
               ▼
          InceptionV3
               │
               ▼
       Tumor Classification
               │
               ▼
        Performance Analysis
```

---

## 📂 Dataset

The MRI dataset is organized into training and testing directories.

```text
MRI Dataset
│
├── Training
│   ├── Class 1
│   ├── Class 2
│   └── ...
│
└── Testing
    ├── Class 1
    ├── Class 2
    └── ...
```

The project uses MRI images for training and evaluating the deep learning models.

---

## 🤖 Models Used

### 1. U-Net

U-Net is used for **image segmentation**. It follows an encoder-decoder architecture and is particularly useful for identifying specific regions within medical images.

### 2. EfficientNetB3

EfficientNetB3 is a convolutional neural network architecture that provides a strong balance between computational efficiency and model performance.

### 3. ResNet50

ResNet50 uses residual connections that help train deeper neural networks effectively.

### 4. InceptionV3

InceptionV3 uses multiple convolutional operations at different scales to extract rich visual features.

---

## 📊 Model Performance

The models were trained and evaluated using the MRI dataset.

The obtained results showed that **EfficientNetB3 and ResNet50 achieved approximately 99% accuracy**, while **InceptionV3 achieved approximately 89% accuracy** in the conducted experiments.

> Performance can vary depending on dataset splits, preprocessing, hyperparameters, and training configuration.

---

## 🔬 Methodology

```text
1. Collect MRI Images
        ↓
2. Preprocess Images
        ↓
3. Split Dataset
        ↓
4. Perform Tumor Segmentation using U-Net
        ↓
5. Train Transfer Learning Models
        ↓
6. Evaluate Models
        ↓
7. Compare Performance
        ↓
8. Select the Better Performing Model
```

---

## 📈 Evaluation Metrics

The project can evaluate model performance using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

These metrics help determine how effectively the models classify MRI images.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd brain-tumor-detection
```

### 2. Install Dependencies

```bash
pip install tensorflow
pip install keras
pip install opencv-python
pip install pillow
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

### 3. Prepare Dataset

Place the MRI dataset in the appropriate training and testing directories.

Example:

```text
dataset/
├── Training/
└── Testing/
```

### 4. Run the Project

Open the project notebook in:

**Google Colab / Jupyter Notebook**

and execute the cells sequentially.

---

## 🔮 Future Enhancements

* 🧠 Improve tumor segmentation accuracy
* 🤖 Experiment with additional CNN architectures
* 📱 Develop a web-based prediction interface
* ☁️ Deploy the trained model online
* 📊 Add interactive visualization
* 🔬 Use larger and more diverse MRI datasets
* ⚕️ Perform extensive clinical validation
* 🖼️ Display segmentation results alongside predictions

---

## 📸 Screenshots

Add screenshots of:

* MRI dataset
* Image preprocessing
* U-Net segmentation results
* Model training graphs
* Confusion matrix
* Classification results
* Prediction output

---

## ⚠️ Disclaimer

This project is intended for **academic, educational, and research purposes**. It should not be used as a standalone medical diagnostic system. Medical decisions should always be made by qualified healthcare professionals.

---

## 📄 License

This project is developed for academic and educational purposes.

---

## 🙌 Acknowledgements

* TensorFlow
* Keras
* Scikit-learn
* OpenCV
* Google Colab
* Medical imaging and deep learning research community
* OpenAI ChatGPT
