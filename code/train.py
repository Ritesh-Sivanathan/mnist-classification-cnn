import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocessor import process_image
import os

# ------------ processing image ------------

path_list = []

for path in os.listdir('..numbers/mnist_png/Hnd'):
    path_list.append(path)

processed_images, labels = process_image(path_list)

# -------------------------------------------

dataset = tf.data.Dataset.from_tensor_slices((processed_images, labels))
batch_size = 3  # Adjust batch size as needed

dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=len(processed_images))

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape = (28,28,1)),  ### h,w,channels
    MaxPooling2D((2,2,)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='softmax')
])

model.compile( # https://keras.io/api/optimizers/adam/
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
) 

dataset = dataset.shuffle(len(processed_images))
valsize = int(0.2*len(processed_images))
training_data = dataset.skip(valsize)
val_dataset = dataset.take(valsize)

history = model.fit(training_data, epochs=50, validation_data=val_dataset)
test_loss, test_acc = model.evaluate(dataset)

print(f'ACCURACY: {test_acc}')

model.save('../model/mnist_model.h5')
