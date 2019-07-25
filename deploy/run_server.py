#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/07/17 13:56:08

@author: Changzhi Sun
"""
import sys
sys.path.append("..")
import configargparse
import flask
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.model import CNN

app = flask.Flask(__name__)

# Initialize our Flask application and the PyTorch model.
model = None
device = None
idx2label = None

def load_model(opt):
    global model
    global device
    global idx2label

    print('Loading checkpoint from %s' % opt.model)
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    print('Loading vocab from checkpoint at %s' % opt.model)
    vocab = checkpoint['vocab']
    idx2label = {v : k for k, v in vocab.items()}

    print('Building model...')
    if opt.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", opt.gpu)
    model = CNN(len(vocab))

    # end of patch for backward compatibility
    print("Loading model parameters from checkpoint...")
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

@app.route("/predict/<version>", methods=["POST"])
def predict(version):
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("mat") and flask.request.files.get("top_num"):
            if version != opt.version:
                print("Version of model (%s) not found !" % version)
                data['help'] = "Version of model (%s) not found !" % version
                return flask.jsonify(data)
            mat = flask.request.files["mat"].read()
            mat = np.frombuffer(mat, np.float32)
            mat = torch.Tensor(mat)
            mat = mat.view(-1, 200, 310)
            mat = mat.to(device)

            top_num = flask.request.files["top_num"].read()
            top_num = int(top_num.decode("utf8"))
            
            #  Classify the input image and then initialize the list of predictions to return to the client.
            preds = F.softmax(model(mat), dim=1).mean(dim=0)

            results = torch.topk(preds.cpu(), k=top_num, dim=0)

            data['results'] = []

            #  Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0], results[1]):
                label_name = idx2label[label.item()]
                r = {"name": label_name, "score": float(prob)}
                data['results'].append(r)

            #  Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

def _get_parser():
    parser = configargparse.ArgumentParser(description="run_server.py")
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', required=True,
                help="Path to model .pt file(s). ")
    group.add('--gpu', '-gpu', type=int, default=-1,
              help="Device to run on")
    group = parser.add_argument_group('SERVER')
    group.add('--host', '-host', type=str, default="127.0.0.1",
                help="The host url")
    group.add('--port', '-port', type=int, default=5000,
                help="The port")
    group.add('--version', '-version', type=str, default="v0",
                help="The version of model")
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    for key, value  in vars(opt).items():
        print("{0} = {1}".format(key, value))
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model(opt)
    #  app.debug = True
    app.run(host=opt.host, port=opt.port)
