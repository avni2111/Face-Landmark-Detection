#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ## Data Cleaning: selecting those images that have 15 facial keypoints

# In[2]:


facial_keypoints = pd.read_csv( "facial_keypoints.csv")
num_missing_keypoints = facial_keypoints.isnull().sum( axis=1 )
all_keypoints_present_ids = np.nonzero( num_missing_keypoints == 0 )[ 0 ]

# face_images.npz 
d = np.load( "face_images.npz")
dataset = d[ 'face_images' ].T
dataset = np.reshape( dataset , ( -1 , 96 , 96 , 1 ) )

images = dataset[  all_keypoints_present_ids , : , : , : ]
keypoints = facial_keypoints.iloc[ all_keypoints_present_ids , : ].reset_index( drop=True ).values

x_train, x_test, y_train, y_test = train_test_split( images , keypoints , test_size=0.3 )

# Save all the processed data.
np.save( "processed_data/x_train.npy" , x_train )
np.save( "processed_data/y_train.npy" , y_train )
np.save( "processed_data/x_test.npy" , x_test )
np.save( "processed_data/y_test.npy" , y_test )


# In[3]:


df = pd.read_csv( "cleaned_facial_keypoints.csv")


# In[4]:


df.head()


# In[5]:


# Load the dataset (make sure to provide the correct path)
df = pd.read_csv("cleaned_facial_keypoints.csv")

# Display the column names
print("Column names:", df.columns)

# Optionally, display the first few rows to inspect the data
print("Dataset preview:")
df.head()


# In[6]:


# Load the dataset
file_path = "cleaned_facial_keypoints.csv"
df = pd.read_csv(file_path)

# Display the column names and a preview of the data
print("Column names:", df.columns)
print("Dataset preview:")
print(df.head())


# In[7]:


# Load the .npz file
npz_file = np.load("face_images.npz")  # Replace with the correct path

# Display the keys as a list
keys = list(npz_file.keys())
print("Keys in the .npz file:", keys)



# In[9]:


import matplotlib.pyplot as plt
# Load the .npz file
npz_file = np.load("face_images.npz")  # Replace with the correct file path

# Retrieve the data under the 'face_images' key
face_images = npz_file['face_images']

# Inspect the shape of the array
print("Shape of 'face_images' array:", face_images.shape)

# Visualize a sample image (assuming the images are 96x96)
plt.imshow(face_images[0], cmap='gray')  # Display the first image
plt.title("Sample Face Image")
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load("face_images.npz")

# List keys in the file
print("Keys in .npz file:", data.files)

# Load the images from the correct key
images = data['face_images']  # Replace 'face_images' if needed
print(f"Original images shape: {images.shape}")

# Transpose the images to correct the shape
images = images.transpose(2, 0, 1)  # Move 7049 to the first axis
print(f"Corrected images shape: {images.shape}")  # (7049, 96, 96)

# Display the first 5 images
for i in range(5):
    plt.imshow(images[i], cmap='gray')  # Use 'gray' for grayscale
    plt.title(f"Image {i + 1}")
    plt.axis('off')
    plt.show()


# In[ ]:




