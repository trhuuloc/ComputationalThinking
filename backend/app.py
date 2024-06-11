from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def predict(img, model):
    img = np.array(img)
    img = cv2.resize(img, (64, 64))
    image = img.astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted

def load_model():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=2, bias=True)
    )
    model.load_state_dict(torch.load('ct_new_epoch_15.pth', map_location=torch.device('cpu')))
    return model.eval()

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model
model_path = str('best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")

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
        classifier = predict(pil_cropped_image, classifier_model)
        buffered = io.BytesIO()
        pil_cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        pred_label = 'fake' if classifier else 'real'
        
        faces.append({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'cropped_image': img_str,
            'classifier': pred_label
        })

    return jsonify({'faces': faces})

if __name__ == '__main__':
    app.run(debug=True)
