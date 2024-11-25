#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = "cleaned_facial_keypoints.csv" 
df = pd.read_csv(file_path)


print(df.head())


# ## Heatmap of Landmark distribution

# In[2]:


# Extract X and Y coordinates for all landmarks
x_coords = df.iloc[:, 0::2].values.flatten() 
y_coords = df.iloc[:, 1::2].values.flatten()  

# Plot heatmap
plt.figure(figsize=(6, 6))
plt.hist2d(x_coords, y_coords, bins=50, cmap='hot')
plt.title("Heatmap of Landmarks Distribution")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.colorbar(label="Frequency")
plt.show()


# ## Scatter Plot for single face 

# In[3]:


# Select the first row (example face) and reshape it into (num_landmarks, 2)
landmarks = df.iloc[0].values.reshape(-1, 2)  # Reshape to (x, y) pairs

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='o')
plt.title("Landmarks for a Single Face")
plt.gca().invert_yaxis()  # Invert Y-axis (image origin is usually top-left)
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.show()


# ## Pair Plot for Selected Landmarks

# In[5]:


# Select a subset of columns (e.g., first 4 landmarks: x1, y1, x2, y2)
subset = df.iloc[:, :8]  # Adjust the range for more landmarks

# Rename columns for better visualization
subset.columns = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

# Create pair plot
sns.pairplot(subset)
plt.suptitle("Pair Plot of Selected Landmarks", y=1.02)
plt.show()


# ## Missing data heatmap

# In[8]:


# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()


# ## Getting accuracy of our model

# In[18]:


import numpy as np

# Load the image data from .npz
images_data = np.load('face_images.npz')
X_images = images_data['face_images']  # Replace 'images' with the correct key from the .npz file


# In[23]:


import numpy as np

# Load the split data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Print shapes to confirm they are loaded correctly
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# In[24]:


# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images to match input dimensions of the model
X_train = X_train.reshape(-1, 96, 96, 1)  # Assuming images are 96x96 grayscale
X_test = X_test.reshape(-1, 96, 96, 1)


# In[25]:


from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('face_landmark_model.keras')

print("Model loaded successfully!")


# In[30]:


import tensorflow as tf
from tensorflow.keras import backend as K

def smooth_l1_loss(y_true, y_pred):
    """
    Smooth L1 Loss function (Huber Loss)
    Args:
        y_true: Ground truth values (real landmark coordinates).
        y_pred: Predicted values (predicted landmark coordinates).
    Returns:
        Smooth L1 loss value.
    """
    diff = K.abs(y_true - y_pred)
    condition = K.less(diff, 1.0)  # Threshold for switching between L1 and L2 loss

    # If the difference is less than 1.0, use L2 loss (squared error).
    loss = K.switch(condition, 0.5 * K.square(diff), diff - 0.5)
    return K.mean(loss)



# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Example CNN Model for Landmark Detection
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(30)  # Output layer for 15 (x, y) landmarks = 30 values
])

# Compile the model with Smooth L1 Loss
model.compile(optimizer=Adam(), loss=smooth_l1_loss, metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[32]:


# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"Test Loss (Smooth L1 Loss): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")

