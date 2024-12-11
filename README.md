# CV_with_ResNet_YOLOv5

## **Table of Contents**
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Overview](#project-overview)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contact](#contact)

---

## **Introduction**

This repository contains solutions to the project for the **ECMM426 Computer Vision** module. It demonstrates the implementation of key computer vision techniques, including:
- Gradient computation
- Bag-of-Visual-Words spatial histograms
- Rotation matrix computation
- CNN training and testing (ResNet)
- YOLOv5 integration for object detection and bounding box processing.

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
- Required folders:
  - `EXCV10/`: Training and testing dataset for CNN.
  - `data/`: Image files for other operations.
  - `MaskedFace/`: YOLOv5-related data.
  - `Answers/`: Directory to save outputs.
 
### Dataset
The **EXCV10 dataset** used for training and testing the CNN can be downloaded from the following link:
[Download EXCV10.zip](https://empslocal.ex.ac.uk/people/staff/ad735/ECMM426/EXCV10.zip)

---

## **Project Overview**

The project includes seven tasks with clearly defined objectives, inputs, and outputs:
1. **Gradient Magnitude and Direction**: Compute gradient magnitude and direction for an input image using custom filters.
2. **Bag-of-Visual-Words Spatial Histograms**: Generate spatial histograms of Bag-of-Visual-Words for divided image regions.
3. **Rotation Matrix Computation**: Compute a rotation matrix for a given set of points and a rotation angle.
4. **CNN Model Training**: Train a ResNet-based CNN on the provided dataset.
5. **CNN Model Testing**: Evaluate the CNN model, computing accuracy and predictions.
6. **Confusion Matrix**: Compute and visualize the confusion matrix for the CNN's predictions.
7. **YOLOv5 Workflow**: Process bounding box data for object detection, including MAPE score calculation.

---

## **Usage Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/ECMM426-Computer-Vision-Coursework.git
   
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
3. ### **Download the EXCV10 dataset**

- [Download the dataset from here](https://empslocal.ex.ac.uk/people/staff/ad735/ECMM426/EXCV10.zip).
- Extract the dataset into the root directory of the repository.

4. Run specific answers:
- Replace answer_1.py with the relevant script for the task
   ```bash
   python answer_1.py

4. Outputs will be saved in the Answers/ folder.

## **Project Structure**

- **EXCV10/**: Training and testing dataset for CNN (ResNet).
- **data/**: Image files for rotation matrix and gradient computation.
- **MaskedFace/**: YOLOv5-related data for object detection tasks.
- **Answers/**: Directory for saving outputs (e.g., `.npy`, `.png`, `.pth` files).
- **scripts/**: Python scripts for each solution.

---

## **Results**

### Task Outputs:

1. **Gradient Magnitude and Direction**: 
   - Computed gradient magnitude and direction stored as `.npy` files.

2. **Bag-of-Visual-Words Spatial Histograms**: 
   - Spatial histograms and visualizations saved as `.npy` and `.png` files.

3. **Rotation Matrix Computation**: 
   - Rotation matrix saved as a `.npy` file.

4. **CNN Model Training**:
   - A **custom, minimal ResNet** architecture was used as specified in the project constraints:
     - `ResNet(block=BasicBlock, layers=[1, 1, 1], num_classes=num_classes)`
     - **BasicBlock**: Simplified residual block with fewer layers.
     - **layers=[1, 1, 1]**: Indicates three stages, each with a single residual block, making this model significantly smaller than standard ResNet variants (e.g., ResNet-18).
   - The performance improvement was achieved through **hyperparameter tuning**, including adjustments to:
     - Number of epochs
     - Learning rate
     - Optimizer type
     - Learning rate scheduler
     - Data augmentation and regularization techniques
   - **No additional layers or depth** were added to the ResNet architecture, as per the project constraints.
   - The best-performing model weights were saved as `data/weights_resnet.pth`.
   - The trained model achieved an accuracy of **72.35%** on the validation set.

5. **CNN Model Testing**: 
   - Predicted labels and accuracy metrics were saved as `.npy` files.

6. **Confusion Matrix**:
   - Confusion matrix and heatmap saved as `.npy` and `.png` files.

7. **YOLOv5 Workflow**:
   - **Objective**:
     - Implement a 3-class (4-class including background) masked face detector to identify:
       - Faces wearing masks correctly (`with_mask`)
       - Faces not wearing masks (`without_mask`)
       - Faces wearing masks incorrectly (`mask_weared_incorrect`)
   - **Functionality**:
     - The YOLOv5 model processes images from the `MaskedFaceTestDataset`, detecting and classifying faces into the specified categories.
     - The function `count_masks(dataset)` counts the number of faces in each class for every image in the dataset.
     - Outputs a 2D numpy array of shape \(N \times 3\), where \(N\) is the number of images, with counts of faces in the three categories for each image.
   - **Performance**:
     - The YOLOv5 model achieved a MAPE score of **13.36%**.
     - Results, including bounding box data and MAPE scores, were saved as `.npy` files in the `Answers/` folder.
