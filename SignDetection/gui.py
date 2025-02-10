import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('sign_language_model_with_opencv.h5')

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Sign Language Detector')
top.configure(background='#CDCDCD')

# Labels to display the result
label1 = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
sign_image = Label(top)

def Detect(file_path):
    """Detect the sign from the uploaded image."""
    global label_packed
    image = Image.open(file_path)
    image = image.resize((64, 64))  # Match the input size of your model
    image = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(image)
    predicted_class = int(np.argmax(prediction))  # Get the class with the highest probability
    confidence = prediction[0][predicted_class]  # Confidence of the prediction

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
    label1.configure(foreground="#011638", text=f"Predicted Sign: {predicted_class} (Confidence: {confidence:.2f})")

def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    """Upload an image and display it in the GUI."""
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error: {e}")

upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
heading = Label(top, text="Sign Language Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
