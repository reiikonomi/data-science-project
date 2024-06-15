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

## Handling Overfitting and Underfitting

### Resolution Strategies
   
1. **Early Stopping**:
   - Monitored validation loss and stopped training when validation metrics stopped improving to prevent overfitting.

2. **Dropout Regularization**:
   - Used dropout layers to randomly drop a fraction of neurons during training to reduce overfitting.

## Examples of Diagrams Showing Overfitting

<!-- ![Overfitting Diagram](overfitting_diagram.png) -->

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
