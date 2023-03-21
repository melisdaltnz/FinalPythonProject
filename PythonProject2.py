import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
import pathlib
import csv
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import json
import pandas as pd
import csv



print(" Welcome to AI! With that program, you can create a AI supervised learning model system that can analyze and evaluate the images and label and match the images with labels!")


#########################GIVEN CODE###################################################
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (evaluating_images, evaluating_images_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag' , 'Ankle boot']    
plt.figure()
plt.imshow(train_images[2])
plt.colorbar()
train_images_re = train_images.reshape(60000,28,28, 1)

######################INPUT-HIDDEN-OUTPUT LAYERS#######################################
def build_model(num_layers, model_name):
    model = Sequential(name = model_name)
    model.add(keras.layers.Conv2D(32, (3,3), padding='same', activation= 'relu',input_shape=(28, 28,1)))
    for i in range(num_layers):
        layer_type = input(f"Enter layer type for layer {i+1} (Conv/Pool/Dense): ")
        if layer_type == "Conv":
            filters = int(input("Enter number of filters: "))
            kernel_size = tuple(map(int, input("Enter kernel size (e.g., 3,3): ").split(",")))
            activation = 'relu'
            model.add(Conv2D(filters, kernel_size,padding="same", activation=activation))
        elif layer_type == "Pool":
            pool_size = tuple(map(int, input("Enter pool size (e.g., 2,2): ").split(",")))
            activation = 'relu'
            model.add(MaxPooling2D(pool_size=pool_size))
        elif layer_type == "Dense":
            units = int(input("Enter number of units: "))
            activation = 'softmax'
            model.add(Dense(units, activation=activation))
        else:
            print("Invalid layer type. Please try again.")
            return None
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    return model

#############################TRAINING(COMPILE-FIT)###################

def train_model(model, num_epochs):
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images_re, train_labels, epochs=10, validation_split=0.4)
    model.save('train_model')
    
plt.grid(False)
plt.show(block=True)
        
############################SAVE ################################################

models = []
models_details = []

num_models = int(input("Enter number of models to build: "))

for i in range(num_models):
    model_name = input(f"Enter name for model {i+1}: ")
    num_layers = int(input("Enter number of layers: "))
    model_architecture = build_model(num_layers,model_name)
    models.append(model_architecture)
    num_epochs = int(input("Enter number of epochs to train: "))
    model_details_item = {"Model Name": model_name, "Architecture": model_architecture, "Num Layers": num_layers, "Num Epochs": num_epochs}
    models_details.append(model_details_item)


for i in range(num_models):
    train_model(models[i], models_details[i]["Num Epochs"])

with open("model_details.csv", mode="w", newline="") as csv_file:
    fieldnames = ["Model Name", "Architecture", "Num Layers", "Num Epochs"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(num_models):
        model_name = models[i].name
        num_layers = len(models[i].layers)
        num_epochs = models_details[i]["Num Epochs"]
        writer.writerow({"Model Name": model_name, "Num Layers": num_layers, "Num Epochs": num_epochs})
        

        
        
        