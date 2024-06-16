![Polis Logo](/assets/logopolis.jpg)

# Water Bottle Image Classification Project

## Description

This project aims to classify images of water bottles into three different categories using Convolutional Neural Networks (CNN). We explore three different classifiers: a baseline model with dense layers only, a convolutional model, and a model leveraging a pre-trained MobileNetV2 architecture. The goal is to achieve high accuracy in classifying water bottle images.

### Classifiers Overview

1. **Baseline Model (Dense Layers Only)**:
   - Uses only dense layers for classification.
   - Achieves a baseline validation accuracy of 90%.
   
2. **Convolutional Model**:
   - Incorporates convolutional layers for feature extraction.
   - Achieves an improved validation accuracy of 93%.
   
3. **Pre-trained Model (MobileNetV2)**:
   - Utilizes a pre-trained MobileNetV2 model with transfer learning.
   - Achieves a high validation accuracy of 95%.

### Classifier 1: Baseline Model (Dense Layers Only)

#### Architecture

The baseline model uses a simple neural network architecture consisting solely of dense (fully connected) layers. This model serves as a foundational benchmark to compare the performance of more complex architectures.

- **Input Layer**: Flattens the 2D image input into a 1D array.
- **Hidden Layers**: Includes two dense layers with ReLU activation functions.
  - First Dense Layer: 128 neurons with ReLU activation.
  - Second Dense Layer: 64 neurons with ReLU activation.
- **Dropout Layers**: Added after each hidden layer to reduce overfitting by randomly setting a fraction of input units to zero during training.
  - Dropout rate: 0.5 after the first dense layer and 0.3 after the second dense layer.
- **Output Layer**: A dense layer with a softmax activation function to output class probabilities for the multi-class classification task.

#### Implementation

```python
# Building the model
model = Sequential([
    Flatten(input_shape=(50, 50, 1)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax'),
    
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Performance
Training Accuracy: High, as the model can easily fit to the training data.
Validation Accuracy: 90%, showing some generalization but also indicating room for improvement with more complex models.

### Classifier 2: Convolutional Model

#### Architecture

The convolutional model incorporates convolutional layers which are better suited for image data as they can capture spatial hierarchies in images. This model includes multiple convolutional layers followed by pooling layers, then dense layers for classification.

- **Convolutional Layers**: Extract spatial features from the images.
- **Conv2D Layer 1**: 32 filters, kernel size 3x3, ReLU activation.
- **MaxPooling2D Layer 1**: Pool size 2x2.
- **Conv2D Layer 2**: 64 filters, kernel size 3x3, ReLU activation.
- **MaxPooling2D Layer 2**: Pool size 2x2.
- **Conv2D Layer 3**: 128 filters, kernel size 3x3, ReLU activation.
- **MaxPooling2D Layer 3**: Pool size 2x2.
- **Flatten Layer**: Flattens the 3D feature maps to 1D feature vectors.
- **Dense Layers**: Perform final classification.
- **Dense Layer**: 512 neurons with ReLU activation.
- **Dropout Layer**: 0.5 dropout rate for regularization.
- **Output Layer**: Dense layer with softmax activation.

#### Implementation

```python
# Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Performance
Training Accuracy: Higher than the baseline model due to better feature extraction.
Validation Accuracy: 93%, showing improved generalization and indicating the effectiveness of convolutional layers in capturing image features.

### Classifier 3: Pre-trained Model (MobileNetV2)

#### Architecture
This model leverages transfer learning using the MobileNetV2 architecture, which is pre-trained on a large dataset (ImageNet). Transfer learning helps to benefit from pre-learned features from a large and diverse dataset.

- **Base Model**: MobileNetV2 without the top classification layer.
   - The base model is used as a feature extractor.
   - Layers are frozen to retain pre-trained weights.
- **Custom Top Layers**: Added for the specific classification task.
   - *GlobalAveragePooling2D Layer*: Reduces each feature map to a single value.
   - *Dense Layer*: 512 neurons with ReLU activation.
   - *Dropout Layer*: 0.5 dropout rate for regularization.
   - *Output Layer*: Dense layer with softmax activation.

#### Implementation

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freezing the base model
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Performance
- Training Accuracy: Very high, as the model benefits from pre-learned features.
- Validation Accuracy: 95%, indicating excellent generalization and leveraging the power of transfer learning for image classification.

## Handling Overfitting and Underfitting

### Resolution Strategies
   
1. **Early Stopping**:
   - Monitored validation loss and stopped training when validation metrics stopped improving to prevent overfitting.

2. **Dropout Regularization**:
   - Used dropout layers to randomly drop a fraction of neurons during training to reduce overfitting.

3. **Data Augmentation**:
   - In theory we would have applied augmentation techniques such as rotation, flipping, and zooming to increase dataset diversity.

## Examples of Diagrams Showing Overfitting

- ![Overfitting Diagram Dense Layer](/assets/dense-layer.png)

- ![Overfitting Diagram CNN](/assets/cnn-diagram.png)

- ![Overfitting Diagram Pre-trained](/assets/pre-trained.png)

The diagram illustrates a typical scenario of overfitting where training accuracy continues to improve while validation accuracy plateaus or declines.

## Comparison Between Different Classifiers

| Classifier           | Validation Accuracy | Key Features                       |
|----------------------|---------------------|------------------------------------|
| Baseline Dense       | 92.27%%             | Dense layers only                  |
| Convolutional        | 99.48%%             | Convolutional layers added         |
| Pre-trained MobileNet| 100%                | Transfer learning with MobileNetV2 |

## Conclusions

- **Performance**: The pre-trained MobileNetV2 model outperformed both the baseline and convolutional models, demonstrating the effectiveness of transfer learning in image classification tasks.
  
- **Complexity vs. Performance**: While the convolutional model improved over the baseline, the pre-trained model achieved the highest accuracy with fewer training epochs and parameters.

- **Practical Application**: For tasks with limited training data, leveraging pre-trained models can significantly enhance model performance and reduce the risk of overfitting.

## Contributors

### Massimo Hamzaj
### Gerti Gegollari
### Rei Ikonomi