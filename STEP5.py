

# import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model from disk
model = load_model('./CNN_model.h5')

IMG_WIDTH, IMG_HEIGHT, CHANNELS = 100, 100, 3

# function to load and prepare the image in proper format
def load_image(img_path, size=(IMG_WIDTH, IMG_HEIGHT)):
    img = image.load_img(img_path, target_size=size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

# Load and pre-process an image
img1 = load_image('./Test/Medium/Crack__20180419_06_19_09,915.bmp')
img2 = load_image('./Test/Large/Crack__20180419_13_29_14,846.bmp')

# Predict the class of the image
result1 = model.predict(img1)
result2 = model.predict(img2)

# The output will be an array with 4 values that sum up to 1. 
# Each value represents the probability of the image belonging to a certain class.
print(f"Probability that the image is a Large: {result1[0][0]}")
print(f"Probability that the image is a Medium: {result1[0][1]}")
print(f"Probability that the image is a None: {result1[0][2]}")
print(f"Probability that the image is a Small: {result1[0][3]}")
print(f"Probability that the image is a Large: {result2[0][0]}")
print(f"Probability that the image is a Medium: {result2[0][1]}")
print(f"Probability that the image is a None: {result2[0][2]}")
print(f"Probability that the image is a Small: {result2[0][3]}")

# To get the class that the model thinks is most likely:
class_idx1 = np.argmax(result1[0])
class_idx2 = np.argmax(result2[0])
print(f"The image is most likely a class {class_idx1}")
print(f"The image is most likely a class {class_idx2}")

# Predict the class of the image
result1 = model.predict(img1)
result2 = model.predict(img2)

# Display the images with annotation
fig, ax = plt.subplots(1,2)

# Load original size image for display
display_img1 = image.load_img('./Test/Medium/Crack__20180419_06_19_09,915.bmp')
display_img2 = image.load_img('./Test/Large/Crack__20180419_13_29_14,846.bmp')

# Add images to plot
ax[0].imshow(display_img1)
ax[1].imshow(display_img2)

# Class names
classnames = ['Class A', 'Class B', 'Class C', 'Class D']

# Add annotation
probabilities1 = [f"Large: {result1[0][0]*100.0:.2f}%",
                  f"Medium: {result1[0][1]*100.0:.2f}%",
                  f"None: {result1[0][2]*100.0:.2f}%",
                  f"Small: {result1[0][3]*100.0:.2f}%"]

probabilities2 = [f"Large: {result2[0][0]*100.0:.2f}% ",
                  f"Medium: {result2[0][1]*100.0:.2f}% ",
                  f"None: {result2[0][2]*100.0:.2f}% ",
                  f"Small: {result2[0][3]*100.0:.2f}% "]

for i, prob in enumerate(probabilities1):
    ax[0].annotate(prob, xy=(0.05, 0.95-0.15*i), xycoords='axes fraction', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', fc='black', alpha=0.5))

for i, prob in enumerate(probabilities2):
    ax[1].annotate(prob, xy=(0.05, 0.95-0.15*i), xycoords='axes fraction', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', fc='black', alpha=0.5))

plt.show()