



# Kidney Stone CT Detection

This project focuses on **AI-powered kidney stone detection** using axial CT images. We build a **simple CNN model** to classify CT images into "Stone" or "Non-Stone". The project includes full data visualization, preprocessing, model training, and prediction steps.

---

## **Table of Contents**
1. [Import Libraries](#import-libraries)  
2. [Visualization of Sample Images](#visualization-of-sample-images)  
3. [Class Distribution](#class-distribution)  
4. [ImageDataGenerator](#imagedatagenerator)  
5. [CNN Model](#cnn-model)  
6. [Train the Model](#train-the-model)  
7. [Training Curves](#training-curves)  
8. [Prediction on Test Image](#prediction-on-test-image)  
9. [About Me](#about-me)  

---

## **Import Libraries**
We use the following libraries for data processing, visualization, and model building:
```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
````

---

## **Visualization of Sample Images**

We randomly visualize a few images from the dataset to understand the data:

```python
# Example: Show random images
# show_random_images(dataset_path, "Stone")
# show_random_images(dataset_path, "Non-Stone")
```

---

## **Class Distribution**

We analyze the number of images in each class to check for balance:

```python
# Example: Bar plot for class distribution
# plt.bar(["Stone", "Non-Stone"], [stone_count, nonstone_count])
```

---

## **ImageDataGenerator**

We use `ImageDataGenerator` for image preprocessing and augmentation:

```python
train_gen = ImageDataGenerator(rescale=1/255)
val_gen   = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory("dataset/train", target_size=(224,224), batch_size=32, class_mode="binary")
val_data   = val_gen.flow_from_directory("dataset/validation", target_size=(224,224), batch_size=32, class_mode="binary")
```

---

## **CNN Model**

We define a simple CNN model for binary classification:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## **Train the Model**

We train the CNN on the training dataset and validate on the validation set:

```python
history = model.fit(train_data, validation_data=val_data, epochs=8)
```

---

## **Training Curves**

We visualize training and validation accuracy and loss:

```python
# Plot training curves
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.show()
```

---

## **Prediction on Test Image**

We predict a single test image and visualize the prediction:

```python
img_path = "dataset/test/Stone/sample.jpg"
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array_expanded = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array_expanded)[0][0]
pred_label = "Stone" if pred>0.5 else "Non-Stone"

plt.imshow(img_array)
plt.title(f"Predicted: {pred_label} ({pred:.2f})")
plt.axis("off")
plt.show()
```

---

## **About Me**

Hi! I am [Arif Miah], a passionate AI/ML enthusiast.
I enjoy working on **medical imaging**, **deep learning**, and **AI applications** to solve real-world problems.

---

