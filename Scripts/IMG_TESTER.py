from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

model= load_model("/home/thugyash/ML/Github/Image_Colorizer/Models/model_epoch_29_val_loss_0.01.keras")
def colorize_image(model, input_image_path, output_image_path, target_size=(256, 256)):
       # Load and preprocess the grayscale image
       grayscale_img = Image.open(input_image_path).convert("L")  # Convert to grayscale
       grayscale_resized = grayscale_img.resize(target_size)  # Resize to model input size
       grayscale_array = np.array(grayscale_resized) / 255.0  # Normalize
       grayscale_array = np.expand_dims(grayscale_array, axis=(0, -1))  # Add batch and channel dims
       # Predict the colorized image
       colorized_array = model.predict(grayscale_array)[0]  # Remove batch dim
       colorized_array = (colorized_array * 255).astype("uint8")  # Rescale to 0-255
       # Save the colorized image
       colorized_img = Image.fromarray(colorized_array)
       colorized_img.save(output_image_path)
       print(f"Colorized image saved at {output_image_path}")


if __name__ == "__main__":
    input_image_path = "/home/thugyash/ML/Github/Image_Colorizer/Data/im1.jpeg"  # Path to the grayscale input image
    output_image_path = "/home/thugyash/ML/Github/Image_Colorizer/Data/im3.jpeg"  # Path to save the colorized image

    # Call the colorize_image function
    colorize_image(model, input_image_path, output_image_path)
