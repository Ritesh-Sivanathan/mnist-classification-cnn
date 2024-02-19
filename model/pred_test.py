from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import random
import tensorflow as tf

def random_test_case():

    test_cases = {}
    print(os.getcwd())
    for dir in os.listdir(f'{os.getcwd()}\\numbers\\mnist_png\\Hnd'):
        for index, file in enumerate(os.listdir(f'{os.getcwd()}\\numbers\\mnist_png\\Hnd\\{dir}')):
            if index <= 10:
                test_cases[(os.listdir(f'{os.getcwd()}\\numbers\\mnist_png\\Hnd\\{dir}'))[random.randint(100, 5000)]] = dir[-1] ## avoids using trained on images
                # dir[-1] is the correct answer
    
    return test_cases

test_cases = random_test_case()

# Load the trained model
model_path = '[MODEL FILE PATH]'
test_path = '[MODEL DIR PATH]'
model = load_model(model_path)

# Load and preprocess the input image

os.chdir('[MNIST HND DIRECTORY]')

correct_prediction = 0
incorrect = 0

for file, correct in test_cases.items():

    img_path = f'Sample{correct}\\{file}'
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # predict VVV
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    print(f"The model classified the image {file} as digit: {predicted_class} | Correct answer {correct}")

    if int(predicted_class) == int(correct):
        correct_prediction += 1
    else:
        incorrect += 1
    
    print(f'FINAL ACCURACY : {correct_prediction}/{correct_prediction+incorrect} : {correct_prediction/(correct_prediction+incorrect)}')
        
