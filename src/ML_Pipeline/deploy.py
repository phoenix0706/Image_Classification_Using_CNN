import tensorflow as tf
import numpy as np
from flask import Flask, request,jsonify
import Utils



app = Flask(__name__)

model_path = '../output/cnn-model.h5'
input_image_path = "../output/api_input.jpg"
ml_model = Utils.load_model(model_path)
img_height = 180
img_width = 180
class_names = ['driving_license', 'others', 'social_security']

@app.post("/get-image-class")
def get_image_class():
    try:
        print("Request received at '/get-image-class'")
        # Save and preprocess the image
        image = request.files['file']
        image.save(input_image_path)
        img = tf.keras.utils.load_img(input_image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Make predictions
        predictions = ml_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Convert to JSON-serializable types
        output = {
            "class": class_names[np.argmax(score)],
            "confidence(%)": float(100 * np.max(score))  # Ensure this is a Python float
        }

        return jsonify(output)  # Properly format the output as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
