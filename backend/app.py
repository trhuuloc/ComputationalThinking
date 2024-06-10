from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
import yolov5
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2


def predict(img, model):
    img = np.array(img)
    img = cv2.resize(img, (64, 64))
    image = img.astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)

    outputs = model(image)
    # return values, indices
    _, predicted = torch.max(outputs, 1)
    return predicted

def load_model():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Sequential(
    nn.Dropout(0.2, inplace=True),
    nn.Linear(in_features=1024, out_features=2, bias=True)
    )
    # state_dict = torch.load(r'C:\HocDaiHoc\HK4\ComputationalThinking\backend\shuffle_epoch_18.pt', map_location=torch.device('cpu'))
    model.load_state_dict(torch.load('shufflenet.pt', map_location=torch.device('cpu')))
    return model.eval()

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model
model = yolov5.load('yolov5s.pt')

classifier_model = load_model()


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
        pil_cropped_image = Image.fromarray(cropped_image)
        classifier = predict(pil_cropped_image,classifier_model)
        buffered = io.BytesIO()
        pil_cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        if classifier:
            pred = 'fake'
        else:
            pred = 'real'
        faces.append({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'cropped_image': img_str,
            'classifier': pred
        })

    return jsonify({'faces': faces})

if __name__ == '__main__':
    app.run(debug=True)
