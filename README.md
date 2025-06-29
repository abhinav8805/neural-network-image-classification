# CIFAR-10 Image Classifier using Convolutional Neural Networks (CNN)

This project demonstrates the implementation of an image classification pipeline using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes.

## Dataset

The CIFAR-10 dataset includes 10 classes:
- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Only a subset of the dataset is used for training and testing:
- **Training samples**: 20,000
- **Testing samples**: 4,000

## Model Architecture

The CNN consists of the following layers:
- Conv2D (32 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel, ReLU activation)
- Flatten
- Dense (64 units, ReLU activation)
- Dense (10 units, Softmax activation)

## Training

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Epochs**: 10
- **Metrics**: Accuracy

## Results

After training, the model is evaluated on the test set and achieves the following performance:

![image](https://github.com/user-attachments/assets/e4e55ec6-9fe9-4f11-aa1f-961a9796f3c3)


## Inference

The trained model can predict the class of new images using OpenCV for preprocessing and matplotlib for visualization.

Example:
```python
img = cv.imread('gadi.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
prediction = model.predict(np.array([img]) / 255.0)
class_index = np.argmax(prediction)
print(f"Predicted class: {class_names[class_index]} with confidence {np.max(prediction) * 100:.2f}%")
