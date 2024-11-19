import os
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

def Model(input_shape=(256, 256, 1)):  # Ensuring grayscale input
    inputs = layers.Input(shape=input_shape)

    # First Convolution Block
    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPool2D()(conv1)

    # Second Convolution Block
    conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = layers.MaxPool2D()(conv2)

    # Third Convolution Block
    conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D()(conv3)

    # Fourth Convolution Block
    conv4 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = layers.MaxPool2D()(conv4)

    # Fifth Convolution Block
    conv5 = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(conv5)

    # Expanding Layers (Upsampling)
    up6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    concat6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(concat6)
    conv6 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv6)

    up7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv6)
    concat7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(concat7)
    conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv7)

    up8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv7)
    concat8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(concat8)
    conv8 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv8)

    up9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv8)
    concat9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(concat9)
    conv9 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    # Output Layer (RGB)
    output = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=output)
    return model

model = Model(input_shape=(256, 256, 1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

def load_img_in_pairs(grayscale_dir,colourized_dir,target_size=(256, 256)):
    grayscale_image=[]
    colourized_image=[]

    grayscale_filenames = os.listdir(grayscale_dir)
    for filename in grayscale_filenames:

        grayscale_path=os.path.join(grayscale_dir,filename)
        grayscale_img=Image.open(grayscale_path).convert('L')
        grayscale_resized=grayscale_img.resize(target_size)
        grayscale_array=np.array(grayscale_resized)/255.0

        colourized_path=os.path.join(colourized_dir,filename)
        colourized_img=Image.open(colourized_path).convert('RGB')
        colorized_resized=colourized_img.resize(target_size)
        colourized_array=np.array(colourized_img)/255.0

        grayscale_image.append(grayscale_array)
        colourized_image.append(colourized_array)

    return np.array(grayscale_image),np.array(colourized_image)


grayscale_images,colourized_images = load_img_in_pairs(
    "/home/thugyash/ML/Github/Image_Colorizer/Data/Train/GR",
    "/home/thugyash/ML/Github/Image_Colorizer/Data/Train/Color"
)

from tensorflow.keras.callbacks import ModelCheckpoint

# Define where to save the checkpoints
checkpoint_dir = "/home/thugyash/ML/Github/Image_Colorizer/Models/"
checkpoint_filepath = checkpoint_dir + "model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5"

# Create a callback to save the model at checkpoints
model_checkpoint = ModelCheckpoint(
    filepath="/home/thugyash/ML/Github/Image_Colorizer/Models/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1,

)

# Train the model with the callback
history = model.fit(
    grayscale_images,          # Input data
    colourized_images,          # Target data
    validation_split=0.2,      # Use 20% of the data for validation
    steps_per_epoch=25,
    epochs=50,                 # Number of epochs
    batch_size=15,             # Batch size
    callbacks=[model_checkpoint]  # Include the checkpoint callback
)

# Save the trained model
try:
    model.save('Models/model1.h5')
    # model.save('/home/thugyash/ML/Github/Image_Colorizer/Models/model.keras')
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
