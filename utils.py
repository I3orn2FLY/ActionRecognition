import torch
import torch.nn as nn
import numpy as np
import os
import sys

from config import *
sys.path.append(os.path.join(PROJECT_DIR, "training"))
from models import LSTM, GRU, NN


def predict(model, X, device):
    inp = torch.Tensor(X).to(device)
    out = model(inp)
    out = out.cpu().numpy()
    out = np.argmax(out, axis=1)

    return out

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def get_model(model_name, device):
    if model_name == "LSTM":
        model = LSTM(input_size=NUM_FEAT, hidden_size=500, output_size=len(classes), num_layers=2, bi=False).to(device)
    elif model_name == "BiLSTM":
        model = LSTM(input_size=NUM_FEAT, hidden_size=500, output_size=len(classes), num_layers=2, bi=True).to(device)
    elif model_name == "GRU":
        model = GRU(input_size=NUM_FEAT, hidden_size=500, output_size=len(classes), num_layers=2, bi=False).to(device)
    elif model_name == "BiGRU":
        model = GRU(input_size=NUM_FEAT, hidden_size=500, output_size=len(classes), num_layers=2, bi=True).to(device)
    elif model_name == "NN":
        model = NN(input_size=NUM_FEAT * SEQ_LENGTH, output_size=len(classes)).to(device)

    if os.path.exists(os.sep.join([WEIGHTS_DIR, model_name + ".pt"])):
        model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, model_name + ".pt"])))
    else:
        model.apply(init_weights)
    return model
