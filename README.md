
---

# **Image Colorizer with Deep Learning**  

This project demonstrates the use of deep learning techniques to colorize grayscale images. By leveraging convolutional neural networks (CNNs), the model can generate realistic RGB images from grayscale input.  

---

## **Features**  
- **Colorization**: Automatically adds colors to grayscale images.  
- **Custom Dataset Support**: Trained on paired grayscale and color images.  
- **Model Checkpointing**: Saves the best-performing model at each epoch.  
- **Image Browsing**: Allows users to browse and test new images.  
- **HDF5 Model Saving**: Trained models are saved for later use and deployment.  

---

## **Technologies Used**  
- **Python**: Core programming language.  
- **TensorFlow/Keras**: Used for building and training the deep learning model.  
- **NumPy**: For numerical computations.  
- **Pillow (PIL)**: For image preprocessing and handling.  

---

## **Installation**  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/YourUsername/Image-Colorizer.git  
   cd Image-Colorizer  
   ```  

2. **Set up a virtual environment**:  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # Linux/Mac  
   venv\Scripts\activate     # Windows  
   ```  

3. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

4. **Prepare your dataset**:  
   - Place grayscale images in the `/Data/GR` directory.  
   - Place corresponding color images in the `/Data/RS` directory.  

---

### **Usage**  

### **Rescaling the Images**
Run the rescaler scrip:
```bash  
python Scripts/IMG_RESCALER.py  
```
The rescaled imagles will be saved 

### **Training the Model**  
Run the training script:  
```bash  
python Scripts/IMG_COLOURISER.py  
```  
The model checkpoints will be saved in the `/Models` directory.  

### **Testing the Model**  
1. Use the testing script to browse and colorize images:  
   ```bash  
   python Scripts/IMG_TESTER.py  
   ```  

2. Select a grayscale image from your system to see the colorized output.  

---

## **Directory Structure**  
```plaintext  
Image-Colorizer/  
├── Data/  
│   ├── GR/             # Grayscale images  
│   ├── RS/             # Color images  
├── Models/             # Saved model checkpoints  
├── Scripts/            # Training and testing scripts  
│   ├── IMG_COLOURISER.py  
│   ├── IMG_TESTER.py  
├── README.md           # Project documentation  
├── requirements.txt    # Dependencies list  
```  

---

## **Contributing**  
Contributions are welcome! Feel free to fork the project, make improvements, and submit pull requests.  

---


Let me know if you need changes or additional sections!