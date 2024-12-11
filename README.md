# CV_with_ResNet_YOLOv5

## **Table of Contents**
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Overview](#project-overview)
  - [Answer 1: Gradient Magnitude and Direction](#answer-1-gradient-magnitude-and-direction)
  - [Answer 2: Bag-of-Visual-Words Spatial Histograms](#answer-2-bag-of-visual-words-spatial-histograms)
  - [Answer 3: Rotation Matrix Computation](#answer-3-rotation-matrix-computation)
  - [Answer 4: CNN Model Training](#answer-4-cnn-model-training)
  - [Answer 5: CNN Model Testing](#answer-5-cnn-model-testing)
  - [Answer 6: Confusion Matrix](#answer-6-confusion-matrix)
  - [Answer 7: YOLOv5 Workflow](#answer-7-yolov5-workflow)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

---

## **Introduction**

This repository contains solutions to the coursework for the **ECMM426 Computer Vision** module. Each solution demonstrates the implementation of key computer vision techniques, including gradient computation, Bag-of-Visual-Words, CNN training and testing, and YOLOv5 integration for object detection and bounding box processing.

---

## **Prerequisites**
- Python 3.8 or later
- Required libraries:
  - OpenCV
  - SciPy
  - NumPy
  - Matplotlib
  - PyTorch (for CNN implementation)
  - scikit-learn
  - YOLOv5 repository
- Files in the `EXCV10/` and `Answers/` folders

---

## **Project Overview**

### **Answer 1: Gradient Magnitude and Direction**
- **Objective**: Compute gradient magnitude and direction of an input image.
- **Key Functions**:
  - `compute_gradient_magnitude()`
  - `compute_gradient_direction()`
- **Input**: `shapes.png`
- **Output**: Gradient magnitude and direction saved as `.npy` files.

### **Answer 2: Bag-of-Visual-Words Spatial Histograms**
- **Objective**: Generate spatial histograms of Bag-of-Visual-Words for specified divisions.
- **Key Functions**:
  - `generate_bovw_spatial_histogram()`
  - `plot_divisions_and_histograms()`
- **Input**: `books.jpg`
- **Output**: Histograms and visualizations saved as `.npy` and `.png` files.

### **Answer 3: Rotation Matrix Computation**
- **Objective**: Compute a rotation matrix for a given set of points and angle.
- **Key Function**: `compute_rotation_matrix()`
- **Input**: Points loaded from `data/points.npy` and a rotation angle.
- **Output**: Rotation matrix saved as `.npy`.

### **Answer 4: CNN Model Training**
- **Objective**: Train a ResNet model on the training dataset.
- **Key Function**: `train_cnn()`
- **Input**: Data from `EXCV10/`.
- **Output**: Best-performing model weights saved as `.pth`.

### **Answer 5: CNN Model Testing**
- **Objective**: Evaluate the trained CNN model.
- **Key Function**: `test_cnn()`
- **Input**: Testing data from `EXCV10/`.
- **Output**: Model accuracy and predicted labels saved as `.npy`.

### **Answer 6: Confusion Matrix**
- **Objective**: Compute and visualize the confusion matrix for model predictions.
- **Key Function**: `compute_confusion_matrix()`
- **Input**: True and predicted labels.
- **Output**: Confusion matrix and heatmap saved as `.npy` and `.png`.

### **Answer 7: YOLOv5 Workflow**
- **Objective**: Process bounding box data for object detection.
- **Key Functions**:
  - `to_yolo()`
  - `to_xml()`
  - `convert_to_yolo_labels()`
  - `count_masks()`
- **Input**: Data from the `MaskedFace/` folder.
- **Output**: MAPE score and bounding box data saved as `.npy`.

---

## **Usage Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/ECMM426-Computer-Vision-Coursework.git
