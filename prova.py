import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# --- Leggere il file JSON di configurazione ---
with open("config.json", "r") as f:
    config = json.load(f)

# --- Impostare seed ---
torch.manual_seed(config["general"]["seed"])

# --- Preprocess MONK dataset ---
def preprocess_monk(path):
    data = pd.read_csv(path, sep=r"\s+", header=None)
    targets = data[data.columns[0]].values.astype(float)
    patterns = data.drop(data.columns[0], axis=1)
    patterns = patterns.drop(data.columns[-1], axis=1)
    patterns = patterns.astype(str)
    one_hot_encoding = pd.get_dummies(patterns, prefix=['A1','A2','A3','A4','A5','A6'])
    input_units_number = one_hot_encoding.shape[1]
    X = torch.tensor(one_hot_encoding.values, dtype=torch.float32)
    y = torch.tensor(targets.reshape(-1,1), dtype=torch.float32)
    return X, y, input_units_number

train_X, train_y, input_size = preprocess_monk(config["paths"]["train_data"])
test_X, test_y, _ = preprocess_monk(config["paths"]["test_data"])

# --- Definizione rete ---
class CustomNN(nn.Module):
    def __init__(self, input_size, neurons_per_layer, output_units, activation_hidden, activation_output, init_range):
        super(CustomNN, self).__init__()
        layers = []
        in_size = input_size
        for h in neurons_per_layer:
            fc = nn.Linear(in_size, h)
            nn.init.uniform_(fc.weight, a=init_range[0], b=init_range[1])
            nn.init.uniform_(fc.bias, a=init_range[0], b=init_range[1])
            layers.append(fc)
            if activation_hidden == "tanh":
                layers.append(nn.Tanh())
            in_size = h
        fc_out = nn.Linear(in_size, output_units)
        nn.init.uniform_(fc_out.weight, a=init_range[0], b=init_range[1])
        nn.init.uniform_(fc_out.bias, a=init_range[0], b=init_range[1])
        layers.append(fc_out)
        if activation_output == "tanh":
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = CustomNN(
    input_size=input_size,
    neurons_per_layer=config["architecture"]["neurons_per_layer"],
    output_units=config["architecture"]["output_units"],
    activation_hidden=config["functions"]["hidden"],
    activation_output=config["functions"]["output"],
    init_range=[config["initialization"]["range_min"], config["initialization"]["range_max"]]
)

# --- Loss e ottimizzatore ---
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["training"]["learning_rate"], momentum=config["training"]["momentum"])

# --- Funzione per calcolare accuracy ---
def compute_accuracy(outputs, targets):
    predicted = torch.where(outputs >= 0.5, 1.0, 0.0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / targets.size(0)

# --- Stampare configurazione ---
architecture_summary = [input_size] + config["architecture"]["neurons_per_layer"] + [config["architecture"]["output_units"]]
print("="*60)
print("Training Configuration:")
print("="*60)
print(f"Architecture: {architecture_summary}")
print(f"Hidden activation: {config['functions']['hidden']}")
print(f"Output activation: {config['functions']['output']}")
print(f"Learning rate: {config['training']['learning_rate']}")
print(f"Momentum: {config['training']['momentum']}")
print(f"Epochs: {config['training']['epochs']}")
print(f"Training samples: {train_X.shape[0]}")
print(f"Test samples: {test_X.shape[0]}")
print("="*60)

# --- Training loop con monitoraggio ---
epochs = config["training"]["epochs"]
train_losses, test_losses = [], []
best_train_acc, best_test_acc = 0, 0

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_X)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    train_acc = compute_accuracy(outputs, train_y)
    with torch.no_grad():
        test_outputs = model(test_X)
        test_loss = criterion(test_outputs, test_y)
        test_acc = compute_accuracy(test_outputs, test_y)
    
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    best_train_acc = max(best_train_acc, train_acc)
    best_test_acc = max(best_test_acc, test_acc)

    # stampare ogni epoca
    print(f"Epoch {epoch:4d}: Train Loss = {loss.item():.4f} (Acc: {train_acc:6.2f}%), Test Loss = {test_loss.item():.4f} (Acc: {test_acc:6.2f}%)")

# --- Plot learning curves ---
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()

# --- Migliori risultati ---
print("="*60)
print("BEST Configuration Found:")
print("="*60)
print(f"Neurons per layer: {config['architecture']['neurons_per_layer']}")
print(f"Learning rate: {config['training']['learning_rate']}")
print(f"Momentum: {config['training']['momentum']}")
print(f"Final Train Accuracy: {best_train_acc:.2f}%")
print(f"Final Test Accuracy: {best_test_acc:.2f}%")
print("="*60)
