# Convolutional Neural Networks – MNIST Digit Classification

This folder contains Jupyter Notebooks implementing and comparing multiple neural network architectures for handwritten digit recognition using the MNIST dataset.

## Architectures Included

1. **Logistic Regression** – No hidden layers.
2. **Fully Connected Network** – Two hidden layers with 200 neurons each (ReLU activation).
3. **Simple CNN** – One convolutional layer (32 filters, 5x5) with MaxPooling and a fully connected layer (1024 neurons).
4. **Deeper CNN** – Two convolutional layers (32 and 64 filters) with MaxPooling and a fully connected layer.
5. **CNN with Dropout** – Same as above, with 50% dropout applied to reduce overfitting.

## Results

- Best validation accuracy achieved: ~99%
- Comparisons were made between minibatch sizes (50 vs. 100) and different architectures.
- Balanced accuracy and training times were measured.

## Visualizations

- Filter weights and feature maps before and after ReLU activation.
