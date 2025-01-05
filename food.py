import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow 
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,InputLayer,AveragePooling2D
from keras.layers import BatchNormalization,Dropout,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50,MobileNetV2
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score, recall_score
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.applications.mobilenet_v2 import preprocess_input
import pickle
# Path to the dataset after unzipping
path = '/content/indian-food-images/Indian Food Images/Indian Food Images'

labels = os.listdir(path)
df = pd.DataFrame(columns=['img_path', 'label'])

for label in labels:
    img_dir_path = os.path.join(path, label)
    for img in os.listdir(img_dir_path):
        img_path = os.path.join(img_dir_path, img)
        df.loc[df.shape[0]] = [img_path, label]

# Show the first few rows of the dataframe
df.head()
# shuffling dataset
df = df.sample(frac=1).reset_index(drop=True)
print("Number of images",df.shape[0])
print("Number of labels",df['label'].nunique())

print('There are only 50 images per class')
plt.figure(figsize=(15,5))
plt.bar(x = df['label'].unique(),height=df['label'].value_counts())
plt.xlabel('classes')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.title('Count of each class')
plt.show()
plt.figure(figsize=(12,6))

for i in range(1,3):
    for j in range(1,4):
        plt.subplot(2,3,3*(i-1) + j)
        img_data = cv2.imread(df['img_path'][3*(i-1) + (j-1)])
        image_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB) # BGR to RGB format
        plt.imshow(image_rgb)
        plt.title(df['label'][3*(i-1) + (j-1)])
        plt.axis('off')

plt.subplots_adjust(wspace=0.2,hspace=0.3)
plt.show()
def preprocess(img_paths):
    images = []
    for img_path in img_paths:
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))
        image_normalized = image_resized.astype('float32') / 255
        images.append(image_normalized)
    return np.array(images)
datagen = ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values to [0, 1]
    rotation_range=20,         # Degree range for random rotations
    width_shift_range=0.2,     # Fraction of total width to shift images horizontally
    height_shift_range=0.2,    # Fraction of total height to shift images vertically
    shear_range=0.2,           # Shear angle in counter-clockwise direction
    zoom_range=0.2,            # Range for random zoom
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Strategy for filling in newly created pixels
)

train_dataset = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'img_path':x_train,'label':y_train}),
    x_col='img_path',
    y_col='label',
    target_size=(224, 224),  
    batch_size=32,
    class_mode='categorical', 
    shuffle=True,
    seed=42,
    subset='training'  
)
class_indices = train_dataset.class_indices

# Print the class indices dictionary
print("Class Indices:")
print(class_indices)
index_to_class = {v: k for k, v in class_indices.items()}
print("Index to Class Mapping:")
print(index_to_class)
def build_model(base):
    base_model = base
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        
        Dense(256, activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        
        Dense(128, activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        
        Dense(80, activation='softmax'),  # For binary classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
mobilenet_model = build_model(MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
history_mobilenet = mobilenet_model.fit(
    train_dataset,
    steps_per_epoch=train_dataset.samples // 32,
    epochs=10
)
inceptionv3_model = build_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
history_inception = inceptionv3_model.fit(
    train_dataset,
    steps_per_epoch=train_dataset.samples // 32,
    epochs=10,
)
