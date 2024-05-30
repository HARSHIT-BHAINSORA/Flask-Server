from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Preprocess the input image
def preprocess_image(file):
    img = image.load_img(BytesIO(file.read()), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Apply preprocess_input
    return img_array

# Define the API endpoint

@app.route('/hey', methods=['GET'])
def hey():
    return 'hey'
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Preprocess the image
    img_array = preprocess_image(file)
    
    # Make the prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    print(class_idx)
    class_names = ['Tomato___Late_blight',
    'Tomato___healthy',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Potato___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Tomato___Early_blight',
    'Tomato___Septoria_leaf_spot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry___Leaf_scorch',
    'Peach___healthy',
    'Apple___Apple_scab',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Bacterial_spot',
    'Apple___Black_rot',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Peach___Bacterial_spot',
    'Apple___Cedar_apple_rust',
    'Tomato___Target_Spot',
    'Pepper,_bell___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Late_blight',
    'Tomato___Tomato_mosaic_virus',
    'Strawberry___healthy',
    'Apple___healthy',
    'Grape___Black_rot',
    'Potato___Early_blight',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_',
    'Grape___Esca_(Black_Measles)',
    'Raspberry___healthy',
    'Tomato___Leaf_Mold',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot',
    'Corn_(maize)___healthy']  
    class_names.sort()
    class_precautions  = [
'Use resistant tomato varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Follow good cultural practices, such as proper watering, fertilization, and weed control, to maintain plant health.',
'Maintain proper vineyard management practices, including pruning, training, and pest control.',
'Remove and destroy infected trees, control the insect vector (Asian citrus psyllid), and use disease-free nursery stock.',
'Follow recommended cultural practices, such as crop rotation, proper fertilization, and weed control.',
'Use resistant varieties, avoid overcrowding, improve air circulation, and apply fungicides if necessary.',
'Follow recommended cultural practices, such as crop rotation, proper fertilization, and weed control.',
'Use resistant varieties, practice crop rotation, remove and destroy infected residues, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected residues, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Follow recommended cultural practices, such as pruning, fertilization, and pest control.',
'Use resistant varieties, practice good sanitation, remove and destroy infected leaves and fruits, and apply fungicides if necessary.',
'Use resistant varieties, control the insect vector (whiteflies), remove and destroy infected plants, and apply insecticides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, apply copper-based bactericides if necessary.',
'Use resistant varieties, practice good sanitation, remove and destroy infected fruits and branches, and apply fungicides if necessary.',
'Follow recommended cultural practices, such as proper pruning, fertilization, and pest control.',
'Use resistant varieties, improve air circulation, remove and destroy infected shoots, and apply fungicides if necessary.',
'Use resistant varieties, practice good sanitation, remove and destroy infected branches, and apply copper-based bactericides if necessary.',
'Use resistant varieties, remove and destroy infected fruits and branches, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Follow recommended cultural practices, such as proper watering, fertilization, and pest control.',
'Use resistant varieties, practice good sanitation, remove and destroy infected leaves and clusters, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Use resistant varieties, control the insect vector (aphids), remove and destroy infected plants, and apply insecticides if necessary.',
'Follow recommended cultural practices, such as proper watering, fertilization, and pest control.',
'Follow recommended cultural practices, such as pruning, fertilization, and pest control.',
'Use resistant varieties, practice good sanitation, remove and destroy infected clusters and prunings, and apply fungicides if necessary.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply fungicides if necessary.',
'Follow recommended cultural practices, such as pruning, fertilization, and pest control.',
'Use resistant varieties, practice crop rotation, remove and destroy infected residues, and apply fungicides if necessary.',
'Use resistant varieties, practice good sanitation, remove and destroy infected vines and prunings, and apply fungicides if necessary.',
'Follow recommended cultural practices, such as pruning, fertilization, and pest control.',
'Use resistant varieties, improve air circulation, remove and destroy infected plants, and apply fungicides if necessary.',
'Use resistant varieties, maintain good cultural practices, apply miticides if necessary, and introduce predatory mites or other biological control agents.',
'Use resistant varieties, practice crop rotation, remove and destroy infected plants, and apply copper-based bactericides if necessary.',
'Follow recommended cultural practices, such as proper fertilization, weed control, and pest management.'
]
    
    class_name = class_names[class_idx]
    precaution = class_precautions[class_idx]
    # Return the prediction as a JSON response
    return jsonify({'disease': class_name,'precaution':precaution})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)