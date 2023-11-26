import cv2
import tkinter as tk
from tkinter import filedialog
import dehazer

def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])

    return file_path

# Select an image file interactively
image_path = select_image()

if image_path:
    # Read the selected image
    hazy_image = cv2.imread(image_path)

    # Perform dehazing
    HazeCorrectedImg, HazeMap = dehazer.remove_haze(hazy_image)

    # Display the images
    cv2.imshow('Hazy Image', hazy_image)
    cv2.imshow('Dehazed Image', HazeCorrectedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No image selected.")
