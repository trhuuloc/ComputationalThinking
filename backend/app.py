from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
import yolov5
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model
model = yolov5.load('yolov5s.pt')

classifier_model = models.shufflenet_v2_x1_0()
classifier_model.fc = nn.Sequential(
    nn.Dropout(0.2, inplace=True),
    nn.Linear(in_features=1024, out_features=2, bias=True)
)
state_dict = torch.load(r'D:\Năm 2 - ANTN\Computational_Thinking\ComputationalThinking\backend\shuffle_epoch_18.pt', map_location=torch.device('cpu'))
classifier_model.load_state_dict(state_dict, strict=False)
classifier_model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)
    image_np = np.array(image) 

    results = model(image)

    faces = []
    for pred in results.pred[0]:
        x1, y1, x2, y2, conf, cls = pred
        cropped_image = image_np[int(y1):int(y2), int(x1):int(x2)]
        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cropped_image = cv2.resize(cropped_image, (64, 64))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        
        pil_cropped_image = Image.fromarray(cropped_image)
        input_tensor = transform(pil_cropped_image).unsqueeze(0)
        
        with torch.no_grad():
            classifier_output = classifier_model(input_tensor)
        classifier = 'real' if torch.argmax(classifier_output) == 0 else 'fake'
        # print(classifier)
        buffered = io.BytesIO()
        # pil_cropped_image = Image.fromarray(image_np[int(y1):int(y2), int(x1):int(x2)])
        pil_cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # print(classifier)

        faces.append({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'cropped_image': img_str,
            'classifier': classifier
        })

    return jsonify({'faces': faces})

if __name__ == '__main__':
    app.run(debug=True)
