import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate
import os

def process_image(img_paths: list[str]) -> list[tf.float32, int]:
    
    processed_images = []
    labels = []

    for directory in img_paths:

        for index, unprocessed_img in enumerate(os.listdir(f'../data/mnist/Hnd/{directory}')):
                image = tf.io.read_file(unprocessed_img)
                processed_image = tf.io.decode_image(image, channels=1) # Mnist dataset is greyscale so rather than 3 channels (rgb) you use just one 
                processed_image = tf.image.resize(processed_image, [28, 28]) 
                processed_image = tf.cast(processed_image, tf.float32) / 255.0
                processed_image = tf.image.random_flip_left_right(processed_image, seed=42)
                processed_image = tf.image.random_brightness(processed_image, max_delta=0.2)                

                processed_images.append(processed_image)
                
                labels.append(int(directory[-1])) # Appends last digit of the directory name since that's the corresponding number

    return processed_images, labels


    
        


