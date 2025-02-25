# Home Assignment 2

## Overview
This repository contains four deep learning tasks covering cloud computing concepts, convolution operations, CNN feature extraction, and implementing CNN architectures using TensorFlow and OpenCV.

## Project Structure
- `Q1_cloud_computing.md` - Definitions and comparisons of cloud computing platforms.
- `Q2_convolution_operations.ipynb` - Convolution operations with different strides and padding
- `Q3_cnn_feature_extraction.ipynb` - Edge detection using Sobel filters and pooling operations.
- `Q4_cnn_architectures.ipynb` - Implementation of AlexNet and ResNet architectures.
- `README.md` - Documentation for the assignment.


## Assignment 1: Cloud Computing for Deep Learning
This task covers fundamental cloud computing concepts related to deep learning.
1. Defined **elasticity** and **scalability** in the context of cloud computing.
2. Compared **AWS SageMaker, Google Vertex AI, and Microsoft Azure ML Studio** based on deep learning capabilities.

### Key Concepts
- Cloud computing benefits for deep learning
- Elasticity vs. Scalability
- Comparison of major cloud AI platforms

## Assignment 2: Convolution Operations with Different Parameters
This task involves performing convolution operations on a 5x5 input matrix using a 3x3 kernel with different stride and padding values.
1. Defined a **5x5 input matrix** and a **3x3 kernel**.
2. Performed **convolution** using four different configurations:
   - Stride = 1, Padding = 'VALID'
   - Stride = 1, Padding = 'SAME'
   - Stride = 2, Padding = 'VALID'
   - Stride = 2, Padding = 'SAME'
3. Printed the resulting feature maps for each case.

### Key Concepts
- Convolution operation in deep learning
- Effect of stride and padding on feature maps

## Assignment 3: CNN Feature Extraction with Filters and Pooling
This task is divided into two parts:

### Task 1: Edge Detection Using Sobel Filter
1. Loaded a grayscale image.
2. Applied **Sobel filters** to detect edges in the x-direction and y-direction.
3. Displayed the original image and the filtered images.

### Task 2: Max Pooling and Average Pooling
1. Created a **random 4x4 matrix** as input.
2. Applied a **2x2 max pooling** operation.
3. Applied a **2x2 average pooling** operation.
4. Printed the original matrix, max-pooled matrix, and average-pooled matrix.

### Key Concepts
- Feature extraction using Sobel filters
- Edge detection in images
- Pooling operations for downsampling

## Assignment 4: Implementing and Comparing CNN Architectures
This task is divided into two parts:

### Task 1: Implementing AlexNet
1. Built a simplified **AlexNet architecture** using TensorFlow.
2. Defined layers including **Conv2D, MaxPooling, Dense, and Dropout**.
3. Printed the model summary after defining the architecture.

### Task 2: Implementing a Residual Block and ResNet
1. Defined a **residual block function** with skip connections.
2. Built a **ResNet-like model** with:
   - Initial **Conv2D layer** (64 filters, 7x7 kernel, stride=2).
   - **Two residual blocks**.
   - **Flatten, Dense, and Output layers**.
3. Printed the model summary.

### Key Concepts
- AlexNet architecture for deep learning
- Residual learning and ResNet model
- Skip connections for efficient training

## Student Information
- **Name:** Bharadwaj Ketham  
- **ID number:** 700759639  
- **Course:** Neural Networks and Deep Learning (CS5720)  


## Dependencies
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

## License
This project is for educational purposes only.

