import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
import cv2
import base64

# Initialize the Flask app
app = Flask(__name__)

# Load the TensorFlow Lite model
model_path = "models/quantized_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def imgReader(image):
    # Resize and preprocess the image
    resized_img = cv2.resize(image, (224, 224))
    normalized_img = resized_img.astype(np.float32) / 255.0

    # Prepare the input tensor
    input_tensor = normalized_img[np.newaxis, ...]

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index']) * 100

    return predictions

def preprocess_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file provided."

        image = request.files['image'].read()
        img1 = request.files['image']
        print(img1, type(img1))
        img = preprocess_image(image)

        if img is not None:
            predictions = imgReader(img)

            # Format each prediction value to two decimal places
            formatted_predictions = [f"{pred:.2f}" for pred in predictions[0]]
            print(formatted_predictions, type(formatted_predictions))

            # Encode the uploaded image as base64 and pass it to the template
            uploaded_image = base64.b64encode(image).decode('utf-8')

            # Pass the formatted_predictions list to the template
            a = formatted_predictions[0]
            h = formatted_predictions[1]
            s = formatted_predictions[2]
            return render_template('index.html', uploaded_image=uploaded_image, predictions=[a, h, s])
        else:
            return "Failed to read or process the image."

    return render_template('index.html', uploaded_image=None, predictions=None)

if __name__ == '__main__':
    app.run(debug=True)
