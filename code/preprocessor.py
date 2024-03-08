import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate
import os

def process_image(img_paths: list[str]) -> list[tf.float32, int]:
    
    processed_images = []
    labels = []

    for directory in img_paths:

        # os.chdir(f'{os.getcwd()}\\{directory}')

        os.chdir(os.path.join('..', directory))

        for index, unprocessed_img in enumerate(os.listdir()):
            if index <= 50:
                image = tf.io.read_file(unprocessed_img)
                processed_image = tf.io.decode_image(image, channels=1) # mnist dataset is greyscale so rather than 3 channels (rgb) you use just one 
                processed_image = tf.image.resize(processed_image, [28, 28]) # all images in mnist datset are 28x28 (i believe)
                processed_image = tf.cast(processed_image, tf.float32) / 255.0
                # processed_image = tf.image.random_flip_left_right(processed_image, seed=42) augmentation just cuz, commented for now
                # processed_image = tf.image.random_brightness(processed_image, max_delta=0.2)                

                processed_images.append(processed_image)
                
                labels.append(int(os.getcwd()[-1])) ### appends last digit of the directory name since that's the corresponding number

        os.chdir(os.path.join('..', 'Hnd')) # replace with whatever the actual mnist main directory is

    return processed_images, labels


    
        


