import cv2
import tkinter as tk
from tkinter import filedialog
import dehazer

def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])

    return file_path

def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]

    # Calculate the ratio to maintain the aspect ratio
    ratio = min(max_width / width, max_height / height)

    # Resize the image
    resized_image = cv2.resize(image, (int(width * ratio), int(height * ratio)))

    return resized_image

# Select an image file interactively
image_path = select_image()

if image_path:
    # Read the selected image
    hazy_image = cv2.imread(image_path)

    # Resize the image to fit in the window
    resized_hazy_image = resize_image(hazy_image)

    # Perform dehazing
    HazeCorrectedImg, HazeMap = dehazer.remove_haze(resized_hazy_image)

    # Resize the dehazed image to fit in the window
    resized_dehazed_image = resize_image(HazeCorrectedImg)

    # Display the images
    cv2.imshow('Hazy Image', resized_hazy_image)
    cv2.imshow('Dehazed Image', resized_dehazed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No image selected.")
