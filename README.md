# CNN - CIFAR-10 Image Classification

A Convolutional Neural Network built with TensorFlow and Keras that classifies images from the CIFAR-10 dataset into 10 categories.

## About the Project

CIFAR-10 contains 60,000 color images of size 32x32 across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. This project trains a VGG-style CNN on the dataset and evaluates how well it recognizes each class.

## Features

- Data augmentation (random flip, rotation, zoom and translation) to reduce overfitting
- Three convolutional blocks with Batch Normalization, MaxPooling and Dropout
- Global Average Pooling followed by a fully connected classifier head
- Callbacks: EarlyStopping, ReduceLROnPlateau and ModelCheckpoint
- Training curves for accuracy and loss
- Evaluation with a classification report and confusion matrix
- Sample predictions with confidence scores
- Feature map visualization of the early convolutional layers
- Saves the trained model for reuse


| Stage | Details |
|-------|---------|
| Input | 32 x 32 x 3 image |
| Augmentation | RandomFlip, RandomRotation, RandomZoom, RandomTranslation |
| Block 1 | 2 x Conv2D (32) + BatchNorm + ReLU, MaxPooling, Dropout 0.2 |
| Block 2 | 2 x Conv2D (64) + BatchNorm + ReLU, MaxPooling, Dropout 0.3 |
| Block 3 | 2 x Conv2D (128) + BatchNorm + ReLU, MaxPooling, Dropout 0.4 |
| Head | Conv2D (256), GlobalAveragePooling, Dense (128), Dropout 0.5 |
| Output | Dense (10) with softmax |

## Requirements

- Python 3.9 or higher
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install everything with:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

```bash
python cnn_full.py
```

The CIFAR-10 dataset downloads automatically the first time you run the script. Training is much faster on a GPU. On a CPU, reduce the `EPOCHS` value at the top of the script.

## Output

All results are saved in the `cnn_outputs/` folder:

| File | Description |
|------|-------------|
| `samples.png` | Sample training images |
| `training_curves.png` | Accuracy and loss over epochs |
| `confusion_matrix.png` | Confusion matrix on the test set |
| `predictions.png` | Sample predictions with confidence |
| `feature_maps_*.png` | Feature maps from the first convolutional layers |
| `best_cnn.keras` | Best model during training |
| `cnn_final.keras` | Final trained model |

## Results

Add your test accuracy here after running the script.
