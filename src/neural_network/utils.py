import matplotlib.pyplot as plt

import random
import json

def load_config_json(filepath):
    """Loads the configuration from the specified JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

#can choose between different extraction methods for the weights/biases
def create_random_extractor(method):
    if(method == "standard"):
        def extractor_function():
            return random.uniform(-0.7,0.7)
        return extractor_function
    else:
        raise ValueError("Invalid method.")
    
def data_splitting(X, T, proportions=[1,0,0]):
    if any(prop < 0 for prop in proportions):
        raise ValueError("Le proporzioni devono essere >= 0")
    
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
        raise ValueError("Sum over proportions list is not 1")

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

def set_dict(d,k,v,sep="."):
    keys = k.split(sep)
    for key in keys[:-1]:
        d = d.setdefault(key, {})

    d[keys[-1]] = v





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