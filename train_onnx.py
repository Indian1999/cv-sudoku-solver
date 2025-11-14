import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import prepare_data as prep_data

"""
Older CPU-s have trouble running tensorflow, so there is option to use onnx instead
"""

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1600, 9)
        )

    def forward(self, x):
        return self.layers(x)
    
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1,epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
        print(f"Epoch: {epoch}/{epochs}, Loss: {l.item():.4f}")

    print("Training complete")
    return model

def export_to_onnx(model, path):
    model.eval()
    dummy = torch.randn(1,1,28,28)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

    print(f"ONNX model saved at: {path}")

def main(args):
    x_train, x_val, x_test, y_train, y_val, y_test = prep_data.get_data(data_choice=args['data'], exclude=args['exclude_fonts'])

    x_train = torch.tensor(x_train.transpose(0,3,1,2), dtype=torch.float32)
    x_val = torch.tensor(x_val.transpose(0,3,1,2), dtype=torch.float32)

    y_train = torch.tensor(y_train.argmax(axis=1), dtype=torch.long)
    y_val = torch.tensor(y_val.argmax(axis=1), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args['batch_size'], shuffle=False)

    model = DigitCNN()
    model = train_model(model, train_loader, val_loader, args['epochs'])
    path = args['model_save_fpath']
    if os.path.exists(path):
        # Append the current date and time to the filepath so we don't overwrite a model
        now = datetime.now()
        suffix = now.strftime("%d_%m_%Y_%H_%M_%S")
        path = f"models/model_{suffix}.onnx"

    export_to_onnx(model, path)
    


if __name__ == '__main__':
    # Construct an argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="both", type=str, help="Choose data to use ('mnist', 'fonts', 'both')")
    ap.add_argument("--exclude_fonts", default=True, type=bool, help="Whether or not to exclude fonts like those in 'data/font_exclude/'")
    ap.add_argument("--model_save_fpath", default="models/model.onnx", type=str)
    ap.add_argument("--batch_size", default="128", type=int)
    ap.add_argument("--epochs", default="10", type=int)
    
    args = vars(ap.parse_args())

    main(args)