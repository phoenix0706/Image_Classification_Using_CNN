

# Convolutional Neural Network (Image Classification)

This project implements a Convolutional Neural Network (CNN) for image classification, allowing users to classify images into categories such as driving license, social security, and others. The project includes modular code for training, testing, and deploying the model with a Flask-based API for real-time predictions.

---

## **Overview**

Convolutional Neural Networks (CNNs) are a powerful tool for working with image and video data. They automatically extract meaningful features from images using convolutional operations, simplifying tasks like image classification. This project follows a structured pipeline to build, train, and deploy a CNN model.

---

## **Aim**

- To understand the basic concepts of CNN.
- To develop a CNN model for image classification.
- To deploy the model using Flask for real-time predictions.

---

## **Tech Stack**

- **Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Matplotlib, Flask, pathlib

---

## **Dataset**

The dataset contains images categorized into three classes:
- **Driving License**
- **Social Security**
- **Others**

The images are preprocessed for uniform size and fed into the model.

---

## **Approach**

1. **Data Loading**: Load training and testing data.
2. **Data Preprocessing**: Resize images, normalize data, and prepare batches.
3. **Model Building and Training**: Build a CNN model using TensorFlow and train it on the dataset.
4. **Data Augmentation**: Improve model generalization by applying augmentations.
5. **Deployment**: Serve the model as an API using Flask for real-time predictions.

---

## **Project Structure**

```
├── input/
│   ├── Training_data/         # Training dataset organized by class
│   └── Testing_Data/          # Testing dataset organized by class
├── output/
│   ├── cnn-model.h5           # Saved trained model
│   └── API Input/             # Temporary folder for testing API inputs
├── src/
│   ├── ML_pipeline/
│   │   ├── Preprocess.py      # Preprocessing logic
│   │   ├── Train_Model.py     # Training logic
│   │   ├── Utils.py           # Utility functions (load/save model, constants)
│   │   └── deploy.py          # Flask-based API for predictions
│   └── Engine.py              # Entry point for training, testing, or deployment
├── Convolutional-Neural-Network.ipynb  # Exploratory notebook
├── Model_API.ipynb                     # Deployment-related notebook
├── requirements.txt                    # Library dependencies
└── README.md                           # Documentation
```

---

## **How to Use**

### **Step 1: Setup Environment**
1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   .\env\Scripts\activate   # For Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you're using Python 3.5–3.8, as TensorFlow versions may have compatibility issues with higher versions.

---

### **Step 2: Run the Project**

1. **Training the Model**:
   - Place training images in `input/Training_data/`.
   - Run:
     ```bash
     python src/Engine.py
     ```
   - Select `0` for training:
     ```plaintext
     Train - 0
     ```
   - The trained model will be saved in the `output/` directory.

2. **Testing the Model**:
   - Place testing images in `input/Testing_Data/`.
   - Select `1` for testing:
     ```plaintext
     Predict - 1
     ```

3. **Deploying the Model**:
   - Select `2` for deployment:
     ```plaintext
     Deploy - 2
     ```
   - The Flask API will run on an address like `http://127.0.0.1:5001/`.

---

### **Step 3: Using the API**

- Example Python request:
  ```python
  import requests

  url = 'http://127.0.0.1:5001/get-image-class'
  files = {'file': open('path/to/image.jpg', 'rb')}
  response = requests.post(url, files=files)
  print(response.json())
  ```

- Expected API response:
  ```json
  {
      "class": "others",
      "confidence(%)": 95.23
  }
  ```

---

## **Future Improvements**

1. **Implement Logging**:
   - Introduce logging to capture training metrics, errors, and API request/response details.
   - Use Python's `logging` module to log events like:
     - Training progress and losses
     - API usage
     - Errors in preprocessing or model prediction

   Example:
   ```python
   import logging

   logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(message)s")
   logging.info("Model training started")
   ```

2. **Add Experiment Tracking**:
   - Integrate MLflow or similar tools to track hyperparameters, metrics, and model versions.

3. **Optimize Deployment**:
   - Add Docker support for containerizing the application.
   - Deploy in production environments using Gunicorn and Nginx.

---

## **Takeaways**

This project covers the following:
1. Basics of CNN and related concepts like pooling, padding, and convolution.
2. Hands-on experience with TensorFlow for data augmentation and modeling.
3. Building modular pipelines for training, testing, and deployment.
4. Creating Flask APIs for serving machine learning models.

---

## **References**
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

