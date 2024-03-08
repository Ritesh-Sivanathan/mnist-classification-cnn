# MNIST Convolutional Neural Network (CNN)

## Overview:
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained on a subset of the MNIST dataset consisting of 60,000 training images and tested on 12,000 test images.

![image](https://github.com/Ritesh-Sivanathan/mnist-classification-cnn/assets/82885975/6a832b65-f7a0-4f91-b9a9-12a790ec72db)

## Requirements:
- Python 3.10.0 ( any other version may work, try at your own discretion )
- TensorFlow 2.15.0
- Keras 2.15.0 ( separate from Tensorflow, issue on my side)
- Numpy 1.26.3
  
## Installation:
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Ritesh-Sivanathan/mnist-classification-cnn.git
    ```

2. Navigate to the project directory:
    ```bash
    cd mnist-classification-cnn
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage:

1. Navigate to the code directory
   ```bash
   cd code
   ```

2. Train the model:
    ```bash
    python train.py
    ```
   This will train the CNN model using the MNIST training data.

3. Change to the model directory
   ```bash
   cd ../model
   ```

3. Evaluate the model:
    ```bash
    python evaluate.py
    ```
   This will evaluate the trained model on the MNIST test data and display accuracy metrics.

## Results:
After training and evaluating the model in `model/evalulate.py`, you'll be shown the model accuracy. Feel free to tweak anything and everything to maximize the accuracy.
<br><br> Some suggestions:
- Changing batch_size `line 20 - train.py`
- Changing filter sizes in model `lines [25-34] - train.py`
- Adding more image augmentation `lines [21 - ...] - preprocessor.py`

## Model Architecture:
The CNN architecture used in this project consists of:
- Input layer (1)
- Convolutional layers (3, Conv2D)
- Max pooling layers (2, MaxPooling2D)
- Flatten layer (1)
- Dense layers (2, relu and softmax activations)
- Output layer

## Contributing:
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.