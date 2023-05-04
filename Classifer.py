# Import relevant modules
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Filter out only the ship images (label 8) for train and test data
train_images_ships = train_images[train_labels[:, 0] == 8]
train_labels_ships = train_labels[train_labels[:, 0] == 8]
test_images_ships = test_images[test_labels[:, 0] == 8]
test_labels_ships = test_labels[test_labels[:, 0] == 8]

# Combine ship and non-ship images for training and test data
train_images = np.concatenate((train_images_ships, train_images[:len(train_images_ships)]))
train_labels = np.concatenate((np.ones_like(train_labels_ships), np.zeros_like(train_labels_ships)))
test_images = np.concatenate((test_images_ships, test_images[:len(test_images_ships)]))
test_labels = np.concatenate((np.ones_like(test_labels_ships), np.zeros_like(test_labels_ships)))

# Preprocess the data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Visualize the models accuracy
history = model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_split=0.2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

#Get model predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

#Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

#Visualize confusion matrix using a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-ship', 'Ship'], yticklabels=['Non-ship', 'Ship'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

