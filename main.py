import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('isl_model.h5')

categories = ['1','2','3','4','5','6','7','8','9',
              'A','B','C','D','E','F','G','H','I','J',
              'K','L','M','N','O','P','Q','R','S','T',
              'U','V','W','X','Y','Z']

# Initialize Tkinter
root = tk.Tk()
root.title("Sign Language Prediction")

# Create a label to display the video feed
video_label = tk.Label(root)
video_label.pack()

# Create a label to display the prediction
prediction_label = tk.Label(root, text="Predicted: ", font=("Helvetica", 16))
prediction_label.pack()

# Start real-time video capture
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Preprocess the frame for your model
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    input_frame = cv2.resize(input_frame, (64, 64))  # Resize to model's input size
    input_frame = input_frame.reshape(1, 64, 64, 1) / 255.0  # Add batch dimension and normalize

    # Make prediction
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = categories[predicted_class]

    # Display the prediction
    prediction_label.config(text=f'Predicted: {predicted_label}')

    # Convert the frame to RGB (Tkinter requires RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    frame_photo = ImageTk.PhotoImage(image=frame_image)

    # Update the video label with the new frame
    video_label.config(image=frame_photo)
    video_label.image = frame_photo

    # Call this function again after 10 milliseconds
    root.after(10, update_frame)

# Start the update loop
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture object when the Tkinter window is closed
cap.release()
cv2.destroyAllWindows()
