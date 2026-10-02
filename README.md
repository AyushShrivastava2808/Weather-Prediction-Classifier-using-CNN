# 🌦️ Weather Prediction Classifier

A deep learning-based **image classification system** that identifies weather conditions from images using a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The model is trained on a dataset containing **6,862 weather images across 11 different classes** and predicts the weather category of a given image.

---

## 🚀 Project Overview

Weather conditions can vary significantly in appearance, making image-based classification a useful computer vision problem.

This project uses a custom CNN to learn visual patterns such as:

* Cloud formations
* Snow-covered scenes
* Rain
* Frost
* Hail
* Lightning
* Fog and smog
* Sandstorms
* Rainbows
* Dew
* Glaze

The trained model can then be used to classify new images and can also perform **frame-by-frame weather detection on video input**.

---

## ✨ Features

* 🧠 Custom CNN architecture built using TensorFlow/Keras
* 🌦️ Classification into 11 weather categories
* 🖼️ Image preprocessing and resizing
* 🔢 Pixel normalization
* 📊 Training and validation monitoring
* ⚡ TensorFlow dataset pipeline optimization
* 🎥 Video-based weather prediction using OpenCV
* 📈 Model accuracy and loss visualization
* 💾 Trained model saved in HDF5 format

---

## 🏷️ Weather Classes

The model classifies images into the following 11 categories:

| Class       | Description                   |
| ----------- | ----------------------------- |
| `dew`       | Dew-covered scenes            |
| `fogsmog`   | Foggy or smoggy conditions    |
| `frost`     | Frost-covered environments    |
| `glaze`     | Glazed/icy weather conditions |
| `hail`      | Hail weather                  |
| `lightning` | Lightning conditions          |
| `rain`      | Rainy weather                 |
| `rainbow`   | Rainbow scenes                |
| `rime`      | Rime-covered surfaces         |
| `sandstorm` | Sandstorm conditions          |
| `snow`      | Snowy weather                 |

The class labels are automatically obtained from the dataset directory structure using TensorFlow's `image_dataset_from_directory`.

---

## 🏗️ Model Architecture

The project uses a custom CNN consisting of multiple convolutional and pooling blocks followed by fully connected layers.

### Architecture

```text
Input Image
   │
   ▼
128 × 128 × 3
   │
   ├── Conv2D (64 filters)
   ├── Conv2D (64 filters)
   ├── MaxPooling2D
   │
   ├── Conv2D (128 filters)
   ├── Conv2D (128 filters)
   ├── MaxPooling2D
   │
   ├── Conv2D (256 filters)
   ├── Conv2D (256 filters)
   ├── MaxPooling2D
   │
   ├── Conv2D (512 filters)
   ├── Conv2D (512 filters)
   ├── MaxPooling2D
   │
   ├── Flatten
   │
   ├── Dense (1024)
   ├── Dropout (0.5)
   │
   ├── Dense (512)
   ├── Dropout (0.5)
   │
   └── Dense (11, Softmax)
            │
            ▼
      Weather Prediction
```

The CNN contains four convolutional blocks with 64, 128, 256 and 512 filters respectively, followed by dense layers of 1024 and 512 units and an 11-class softmax output layer.

The model contains approximately **38.77 million trainable parameters**.

---

## 📂 Dataset

The dataset is organized into separate directories for each weather class.

```text
dataset/
├── dew/
├── fogsmog/
├── frost/
├── glaze/
├── hail/
├── lightning/
├── rain/
├── rainbow/
├── rime/
├── sandstorm/
└── snow/
```

### Dataset Statistics

* **Total images:** 6,862
* **Number of classes:** 11
* **Image size:** 128 × 128 pixels
* **Training images:** 5,490
* **Validation/Test pool:** 1,372

The notebook creates an 80/20 training-validation split using a fixed random seed of 123. The validation portion is then divided into validation and test subsets for evaluation.

> **Note:** The original dataset is not included in this repository.

---

## 🔄 Data Preprocessing

Before being passed to the CNN, images are resized to:

```text
128 × 128 × 3
```

During inference, pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
img_array = img_array / 255.0
```

The same input size and normalization approach are used during prediction.

---

## ⚙️ Training Configuration

The model was trained with:

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Image Size        | 128 × 128                       |
| Batch Size        | 32                              |
| Epochs            | 20                              |
| Optimizer         | Adam                            |
| Loss Function     | Sparse Categorical Crossentropy |
| Output Activation | Softmax                         |
| Number of Classes | 11                              |

These settings are directly reflected in the training notebook.

---

## 📊 Model Performance

The model was trained for 20 epochs.

### Final Training Metrics

```text
Training Accuracy:    93.10%
Validation Accuracy:  70.29%
Test Accuracy:        69.64%
```

The recorded test evaluation produced:

```text
Test Accuracy: 0.6964
```

or approximately:

### **69.64% Test Accuracy**

### Training Observation

The training accuracy increased substantially throughout training, reaching approximately 93.1% by epoch 20, while validation accuracy remained around 70%.

This indicates a noticeable **generalization gap**, suggesting that the model learned the training data much more strongly than unseen samples.

This also provides a useful direction for future improvements such as:

* Data augmentation
* Transfer learning
* Batch normalization
* Learning-rate scheduling
* Early stopping
* Better regularization
* Class balancing
* More systematic hyperparameter tuning

---

## 🎥 Video Weather Detection

The project also demonstrates weather classification on video frames using OpenCV.

The inference pipeline:

```text
Video
  ↓
Read Frame
  ↓
Resize to 128 × 128
  ↓
Normalize Pixel Values
  ↓
CNN Prediction
  ↓
Find Highest Probability Class
  ↓
Display Weather Label + Confidence
```

The notebook loads the trained `weather_model.h5` model and processes frames from a video file. The predicted class and confidence score are displayed directly on the video frame.

Example prediction format:

```text
rain : 0.87
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Networks (CNN)

### Data Processing

* NumPy

### Computer Vision

* OpenCV

### Visualization

* Matplotlib

### Development Environment

* Jupyter Notebook
* Google Colab

---

## 📁 Repository Structure

```text
Weather-Prediction-Classifier/
│
├── weather_model.ipynb
├── weather_model_colab.ipynb
├── requirements.txt
└── README.md
```

### Notebook Description

**`weather_model.ipynb`**

Contains the main model workflow and inference implementation, including video-based weather detection.

**`weather_model_colab.ipynb`**

Contains the complete Google Colab training workflow, including:

* Dataset preparation
* Dataset loading
* Train/validation split
* CNN architecture
* Model training
* Evaluation
* Model saving
* Accuracy visualization

The repository currently contains these two notebooks.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/AyushShrivastava2808/Weather-Prediction-Classifier.git
```

Move into the project directory:

```bash
cd Weather-Prediction-Classifier
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Option 1 — Google Colab

Open:

```text
weather_model_colab.ipynb
```

in Google Colab.

Upload or mount the dataset according to the notebook's expected directory structure:

```text
dataset/
├── dew/
├── fogsmog/
├── frost/
├── glaze/
├── hail/
├── lightning/
├── rain/
├── rainbow/
├── rime/
├── sandstorm/
└── snow/
```

Run the notebook cells sequentially to train and evaluate the model.

---

### Option 2 — Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
weather_model.ipynb
```

or:

```text
weather_model_colab.ipynb
```

and execute the cells.

---

## 🎯 Prediction Workflow

For a new image, the model follows these steps:

```text
Input Image
     ↓
Resize → 128 × 128
     ↓
Normalize → Pixel / 255
     ↓
Add Batch Dimension
     ↓
CNN Model
     ↓
Softmax Probabilities
     ↓
Highest Probability
     ↓
Predicted Weather Class
```

---

## 🔮 Future Improvements

The current model provides a strong baseline for weather image classification. Future versions can improve both accuracy and generalization through:

### 1. Transfer Learning

Use pretrained architectures such as:

* MobileNetV2
* EfficientNet
* ResNet50
* InceptionV3

### 2. Data Augmentation

Introduce transformations such as:

* Rotation
* Horizontal flipping
* Zoom
* Translation
* Brightness adjustment

### 3. Regularization

Reduce overfitting using:

* Batch Normalization
* Increased Dropout
* L2 Regularization
* Early Stopping

### 4. Model Optimization

Experiment with:

* Learning-rate scheduling
* Different optimizers
* Hyperparameter tuning
* Class weighting

### 5. Deployment

The trained model could be integrated into:

* Streamlit application
* Flask/FastAPI API
* Web-based weather image classifier
* Mobile application

---

## ⚠️ Limitations

* The model is trained on a fixed set of 11 weather categories.
* Prediction quality depends heavily on the visual similarity between training and real-world images.
* The recorded test accuracy is approximately 69.64%.
* The model shows a gap between training and validation/test performance.
* Real-world weather conditions can contain multiple overlapping visual characteristics that may not correspond to a single class.

---

## 📌 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Image classification
* Convolutional Neural Networks
* TensorFlow/Keras
* Dataset loading with `image_dataset_from_directory`
* Train-validation splitting
* Image preprocessing
* CNN architecture design
* Softmax multiclass classification
* Model training and evaluation
* Overfitting analysis
* Model saving and loading
* OpenCV video processing
* Real-time/frame-based inference

---

## 👨‍💻 Author

**Ayush Shrivastava**

GitHub:
https://github.com/AyushShrivastava2808

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐.
