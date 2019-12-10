import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("../")

from utils import get_model
from config import *




def acc_by_batch_predict(model, X, y, device, batch_size):
    L = X.shape[0]
    right = 0
    for start in range(0, L, batch_size):
        end = start + batch_size if start + batch_size < L else L
        inp = torch.Tensor(X[start:end]).to(device)
        out = model(inp)
        pred = out.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        right += np.sum(y[start:end] == pred)

    return right * 100 / L


def calc_acc(targets, pred):
    return np.sum(targets == pred) * 100 / len(pred)


# TODO
# Make more choosing of models
# Code to compare to KTH benchmarks

def train(X_train, y_train, X_val, y_val, X_test, y_test,
          model, device, model_name,
          num_epoch=50, lr=0.0005, batch_size=4096):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        best_val = acc_by_batch_predict(model, X_val, y_val, device, batch_size)
        print("Validation Accuracy on loaded model:", best_val)
        test_acc = acc_by_batch_predict(model, X_test, y_test, device, batch_size)
        print("Test Accuracy on loaded model:", test_acc)

    losses = []
    for epoch in range(1, num_epoch + 1):
        model.train()
        L = X_train.shape[0]
        right = 0
        epoch_losses = []
        len(range(0, L, batch_size))
        for start in range(0, L, batch_size):
            optimizer.zero_grad()
            end = start + batch_size if start + batch_size < L else L

            inp = torch.Tensor(X_train[start:end]).to(device)
            out = model(inp)
            labels = torch.LongTensor(y_train[start:end]).to(device)
            loss = loss_fn(out, labels)
            loss.backward()

            epoch_losses.append(loss.item())
            optimizer.step()

            pred = out.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)

            right += np.sum(y_train[start:end] == pred)

        losses.append(np.mean(epoch_losses))
        model.eval()
        with torch.no_grad():

            train_acc = right / len(y_train) * 100

            val_acc = acc_by_batch_predict(model, X_val, y_val, device, batch_size)
            test_acc = acc_by_batch_predict(model, X_test, y_test, device, batch_size)

            print("Epoch", epoch,
                  "Train Acc: %.2f" % train_acc,
                  "Val Acc: %.2f" % val_acc,
                  "Test Acc: %.2f" % test_acc)

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), os.sep.join(["..", "weights", model_name + ".pt"]))
                print("Weights Saved")

    return losses





def plot_learning_curve(losses):
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")

    plt.savefig("learning_curve.png")
    plt.show()


if __name__ == "__main__":
    X_train = np.load("../vars/X_train.npy")
    X_test = np.load("../vars/X_test.npy")
    X_val = np.load("../vars/X_val.npy")
    y_train = np.load("../vars/y_train.npy")
    y_test = np.load("../vars/y_test.npy")
    y_val = np.load("../vars/y_val.npy")
    print(X_train.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "LSTM"
    # model_name = "BiLSTM"
    # model_name = "GRU"
    # model_name = "BiGRU"
    # model_name = "NN"

    if model_name == "NN":
        X_train = X_train.reshape(-1, SEQ_LENGTH * NUM_FEAT)
        X_test = X_test.reshape(-1, SEQ_LENGTH * NUM_FEAT)
        X_val = X_val.reshape(-1, SEQ_LENGTH * NUM_FEAT)

    model = get_model(model_name, device)

    losses = train(X_train, y_train, X_val, y_val, X_test, y_test, model, device, model_name,
                   batch_size=1024, lr=0.001, num_epoch=50)

    # plot_learning_curve(losses)
