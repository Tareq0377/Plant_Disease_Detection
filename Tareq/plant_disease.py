import numpy as np
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


default_image_size = tuple((100, 100))
image_size = 0
directory_root = r"../data/PlantVillage"

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

image_list, label_list = [], []
try:
    print("Loading images ...")
    root_dir = listdir(directory_root)
    
    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for plant_disease_folder in plant_disease_folder_list:
            print(f"Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}")
              
            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    
                    label_list.append(plant_disease_folder)
    print("Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

print('image size ',len(image_list))

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)

n_classes = len(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 255.0

print("Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 

img_gen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')) 
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')) 
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
model.add(Flatten()) 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5, name='Dropout_1'))
model.add(Dense(n_classes, activation='softmax'))

opt = Adam(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
model.summary()

# train the network
print("training network...")

history = model.fit_generator(
    img_gen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train)//32,
    epochs=20, verbose=1
    )


#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Both in same figure
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy/Loss')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc','train_loss', 'test_loss'], loc='center left')
plt.show()

print("Model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

