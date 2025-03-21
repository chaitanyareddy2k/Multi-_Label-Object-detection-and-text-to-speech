import flask
from flask import Flask, request, render_template, send_file
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os
import uuid


from gtts import gTTS
from collections import OrderedDict

app = Flask(__name__)


AUDIO_DIR = 'static/audio'
os.makedirs(AUDIO_DIR, exist_ok=True)


def remove_prefix(state_dict, prefix):

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[len(prefix):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


@app.route("/")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        file = request.files['image']
        if not file:
            return render_template('index.html', label="No file uploaded")

        img = Image.open(file)
        img = transforms.functional.resize(img, 250)
        img = transforms.functional.five_crop(img, 224)

        f = lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in img])
        feature = f(img)

        k = lambda norm: torch.stack(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in feature])
        features = k(feature)


        output = model(features)

        output = output.mean(0)

        output = output.numpy().ravel()
        labels = thresh_sort(output, 0.5)

        if len(labels) == 0:
            label = "There are no Pascal VOC categories in this picture"
            tts_text = "No recognizable objects found in the image"
        else:
            label_array = [cat_to_name[str(i)] for i in labels]
            label = "Predictions: " + ", ".join(label_array)
            tts_text = "I detect the following objects: " + ", ".join(label_array)


        unique_filename = f'prediction_audio_{uuid.uuid4()}.mp3'
        audio_path = os.path.join(AUDIO_DIR, unique_filename)


        tts = gTTS(text=tts_text, lang='en')
        tts.save(audio_path)

        return render_template('index.html',
                               label=label,
                               audio_file=f'audio/{unique_filename}',
                               audio_present=True)


def thresh_sort(x, thresh):
    idx, = np.where(x > thresh)
    return idx[np.argsort(x[idx])]


def init_model():
    np.random.seed(2019)
    torch.manual_seed(2019)


    resnet = models.resnet50()
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 20)


    checkpoint = torch.load('model.pth', map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = remove_prefix(state_dict, 'backbone.')

    resnet.load_state_dict(state_dict, strict=False)


    for param in resnet.parameters():
        param.requires_grad = False

    resnet.eval()
    return resnet


if __name__ == '__main__':

    model = init_model()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    app.run(host='0.0.0.0', port=8000)
