import torch

import transformers
from flask import Flask, request, jsonify
from config import *
from utils import model_loading, one_inference_fn

app = Flask(__name__)
device = 'cpu'
path = "saved/sentimental_camembert_model.pth"
model = model_loading(path)
model.to(device)
#tokenizer = transformers.CamembertTokenizer.from_pretrained("camembert-base")

@app.route('/predictions', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        sentence = request.form['sentence']
        category, proba = one_inference_fn(sentence,
                                           model,
                                           tokenizer, 
                                           device=device,
                                           max_length=512)
        proba_str = str(round(100 * proba, 2))+' %'
        return jsonify(category=category, proba=proba_str)
    



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)