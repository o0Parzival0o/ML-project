import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import random
import json
import os

def load_config_json(filepath):
    """Loads the configuration from the specified JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

#can choose between different extraction methods for the weights/biases
def create_extractor(method):
    range = 1.4
    if(method == "random"):
        def random_extractor(fan_in=None, fan_out=None, a=None):
            return random.uniform(-range/2, range/2)
        return random_extractor
    elif(method == "fan_in"):   # Micheli (Slide NN-part2 n.12) 
        def fanin_extractor(fan_in=None, fan_out=None, a=None):
            bound = range * 2. / fan_in
            return random.uniform(-bound, bound)
        return fanin_extractor
    elif(method == "xavier"):   # sigmoid/tanh
        def xavier_extractor(fan_in=None, fan_out=None, a=None):
            bound = np.sqrt(6. / (fan_in - fan_out))
            return random.uniform(-bound, bound)
        return xavier_extractor
    elif(method == "he"):   # ReLu
        def he_extractor(fan_in=None, fan_out=None, a=None):
            bound = np.sqrt(6. / (fan_in * (1 + a**2)))
            return random.gauss(-bound, bound)
        return he_extractor
    else:
        raise ValueError("Invalid method.")
    
def standardization(X):
    eps = 1e-8                              # for non zero std
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + eps
    return mean, std
    
def scaling(X):                        # remember to do the inverse at the end with "inverse_scaling"
    eps = 1e-8                              # for non zero deltaX
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0) + eps
    return X_min, X_max

def inverse_standardization(X, X_mean, X_std):
    return X * X_std + X_mean

def inverse_scaling(X, X_min, X_max):
    return X * (X_max - X_min) + X_min

def data_splitting(X, T, proportions=[1,0,0]):
    if any(prop < 0 for prop in proportions):
        raise ValueError('Elements in "proportions" must be greater or equal than 0')
    
    if sum(proportions) == 1:
        n = len(X)
        X_splits = []
        T_splits = []

        start = 0
        for p in proportions:
            end = start + int(p * n)

            X_splits.append(X[start:end])
            T_splits.append(T[start:end])
            
            start = end

        # Ritorno direttamente tutti i vettori come variabili separate
        return (*X_splits, *T_splits)
    
    else:
        raise ValueError('Sum over "proportions" list is not 1')

def flatten_config(d, parent_key='', sep='.'):

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())

        elif isinstance(v, list):
            items.append((new_key, v))
            
        else:
            items.append((new_key, [v]))
            
    return dict(items)

def set_dict(d, k, v, sep="."):
    keys = k.split(sep)
    for key in keys[:-1]:
        d = d.setdefault(key, {})

    d[keys[-1]] = v

def loguniform(v0, v1):
    return 10 ** np.random.uniform(np.log10(v0), np.log10(v1))

def plot_dataset(X, T, X_test=None):
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i <= j:
                continue

            target_size = T.shape[1]
            
            cols = int(np.ceil(np.sqrt(target_size)))
            rows = int(np.ceil(target_size / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            x_min, x_max = np.min(X[:, i]), np.max(X[:, i])
            y_min, y_max = np.min(X[:, j]), np.max(X[:, j])

            for k in range(target_size):
                r = k // cols
                c = k % cols
                ax = axes[r][c]

                if X_test is not None:
                    ax.scatter(X_test[:,i], X_test[:,j], color='r', edgecolor="none", alpha=0.75, s=10, label='data CUP')
                data = ax.scatter(X[:, i], X[:, j], c=T[:, k], edgecolor="none", alpha=0.75, s=10)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                ax.set_title(f"Output {k}", fontsize=12, fontweight="bold")
                ax.set_xlabel(f"Feature {i}")
                ax.set_ylabel(f"Feature {j}")

                color = fig.colorbar(data, ax=ax, shrink=0.85)
                color.ax.tick_params(labelsize=8)

            plt.tight_layout()
            plt.savefig(f'../../plots/data_plot/plot_{i}_{j}.png', dpi=300)
            plt.close()

def plot_correlation(X, T):

    feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    target_names = [f"Target {i}" for i in range(T.shape[1])]
    
    df_X = pd.DataFrame(X, columns=feature_names)
    corr_features = df_X.corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_features, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix - Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../plots/features_correlation.png', dpi=300)
    plt.show()

    df_T = pd.DataFrame(T, columns=target_names)
    df_combined = pd.concat([df_X, df_T], axis=1)
    corr_full = df_combined.corr()
    
    # Estrai solo correlazioni features-target
    corr_feat_target = corr_full.loc[feature_names, target_names]
    
    # Plot
    plt.figure(figsize=(8, 10))
    sns.heatmap(corr_feat_target, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Correlation: Features - Targets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../plots/features_target_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_features, corr_feat_target

def get_k_fold_indices(n_samples, k_folds):
    folds_size = int(np.floor(n_samples / k_folds))
    folds_remainder = n_samples % k_folds
    
    folds_indexes = []
    start_index = 0
    
    for i in range(k_folds):
        current_fold_size = folds_size + (1 if i < folds_remainder else 0)
        current_end = start_index + current_fold_size
        folds_indexes.append((start_index, current_end))
        start_index = current_end
        
    return folds_indexes


#=============================
# PLOT (da eliminare prima di mandare a Micheli)
#=============================

def gather_weights(nn):
        layers = nn.hidden_layers + [nn.output_layer]
        all_weights = []
        for layer in layers:
            for neuron in layer.neurons:
                all_weights.extend(neuron.weights)
        return all_weights

def plot_network(
    nn,
    figsize=(10, 6),
    weight_scaling=5.0,
    show_bias=True,
    layer_spacing=0.7,
):
    """
    Visualizzazione migliorata della rete neurale:
    - layer più compatti
    - neuroni centrati verticalmente
    - linee con opacità proporzionale al peso
    - nodi più eleganti
    """

    # --- Layer sizes ---
    layer_sizes = [nn.num_inputs] \
                    + [len(l.neurons) for l in nn.hidden_layers] \
                    + [len(nn.output_layer.neurons)]
    n_layers = len(layer_sizes)

    # --- Compute node positions ---
    positions = []
    for i, size in enumerate(layer_sizes):
        x = i * layer_spacing
        if size == 1:
            ys = [0.5]
        else:
            spacing = 0.15  # distanza verticale minima tra neuroni
            total_height = (size - 1) * spacing
            if total_height > 1 - 2 * spacing:
                # troppi neuroni, distribuisco come prima con padding
                padding = spacing
                ys = [padding + j * (1 - 2 * padding) / (size - 1) for j in range(size)]
            else:
                # layer piccolo: centro verticale
                start_y = 0.5 - total_height / 2
                ys = [start_y + j * spacing for j in range(size)]
        positions.append([(x, y) for y in ys])

    # --- Weight stats for scaling ---
    all_w = gather_weights(nn)
    max_abs_w = max((abs(w) for w in all_w), default=1.0)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, (n_layers - 1) * layer_spacing + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # --- Draw edges ---
    layers_obj = nn.hidden_layers + [nn.output_layer]
    for li, layer in enumerate(layers_obj, start=1):
        prev_pos = positions[li - 1]
        cur_pos = positions[li]
        for ni, neuron in enumerate(layer.neurons):
            for pi, w in enumerate(neuron.weights):
                start = prev_pos[pi]
                end = cur_pos[ni]
                norm_w = w / max_abs_w if max_abs_w else 0

                color = (0.22, 0.42, 0.88) if w >= 0 else (0.88, 0.22, 0.22)
                lw = max(0.3, abs(norm_w) * weight_scaling)
                alpha = 0.4 + 0.6 * abs(norm_w)

                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=color,
                    linewidth=lw,
                    alpha=alpha
                )

    # --- Draw nodes ---
    for li, layer_pos in enumerate(positions):
        xs = [p[0] for p in layer_pos]
        ys = [p[1] for p in layer_pos]

        if li == 0:
            # input layer
            ax.scatter(xs, ys, s=180, facecolors="#e8e8e8",
                       edgecolors="#444", linewidths=1.0, zorder=5)
            for i, (x, y) in enumerate(layer_pos):
                ax.text(x - 0.05, y, f"i{i}", ha="right", va="center",
                        fontsize=11, color="#333")
        else:
            # hidden/output layers
            ax.scatter(xs, ys, s=300, facecolors="#ffffff",
                       edgecolors="#333", linewidths=1.2, zorder=6)

            layer_obj = layers_obj[li - 1]
            for ni, (x, y) in enumerate(layer_pos):
                if show_bias:
                    bias = layer_obj.neurons[ni].bias
                    ax.text(x, y + 0.05, f"{bias:.2f}",
                            ha="center", va="center", fontsize=11,
                            color="#1a7f1a", alpha=0.8)

    ax.set_title("Mappa dei pesi e dei bias", fontsize=13)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.07)
    plt.tight_layout()
    plt.savefig("plot.png")

def plot_weight_histogram(nn, bins=40, figsize=(6,3)):
    all_w = gather_weights(nn)
    plt.figure(figsize=figsize)
    plt.hist(all_w, bins=bins, color='steelblue', edgecolor='k', alpha=0.8)
    plt.title("Distribuzione dei pesi")
    plt.xlabel("Valore peso")
    plt.ylabel("Frequenza")
    plt.grid(alpha=0.3)
    plt.show()