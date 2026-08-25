import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Input, Dense,RNN, LSTMCell, GRUCell, GlobalAveragePooling1D, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.datasets import mnist


#imported families architectures
from nns import *
from ncps.wirings import AutoNCP

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. TensorFlow is using CPU.")


# Config
base_model_name = 'Event_based_MNIST'
weights_dir = 'model_weights'
os.makedirs(weights_dir, exist_ok=True)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# ---- Preprocessing functions ----
def binarize(img, threshold=128):
    return (img >= threshold).astype(np.int32)

def sequentialize(img):
    return img.flatten()

def event_based_compression(seq):
    events = []
    prev_val = seq[0]
    t = 1
    for i in range(1, len(seq)):
        if seq[i] == prev_val:
            t += 1
        else:
            events.append((prev_val, t))
            prev_val = seq[i]
            t = 1
    events.append((prev_val, t))
    return events

def normalize_time(events, max_len=256):
    total_time = sum(t for _, t in events)
    scale = max_len / total_time
    return [(val, t * scale) for val, t in events]

def pad_events(events, max_len=256):
    events = events[:max_len]
    if len(events) < max_len:
        events += [(0, 0)] * (max_len - len(events))
    return events

def preprocess_dataset(images, labels, max_len=256):
    event_data = []
    for img in images:
        binary_img = binarize(img)
        seq = sequentialize(binary_img)
        events = event_based_compression(seq)
        events = normalize_time(events, max_len)
        padded = pad_events(events, max_len)
        event_data.append(padded)
    return np.array(event_data, dtype=np.float32), labels

# Apply preprocessing
X_events, y = preprocess_dataset(X, y)

# Wiring
ncp_wiring = AutoNCP(units=64,output_size=10)

# ---- Model builder ----
def build_model(cell_type, input_shape=(256,2), num_classes=10):
    inp = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same', strides=5)(inp)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same', strides=5)(x)
    
    if cell_type == "LSTM":
        x = RNN(LSTMCell(64), return_sequences=False)(x)
    elif cell_type == "GRU":
        x = RNN(GRUCell(64), return_sequences=False)(x)
    elif cell_type == "SDPA-Transformer":
        x = SPDATransformer(embed_dim=64, num_heads=16,ff_dim=32)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "CfC":
        x = CfC(ncp_wiring, return_sequences=False)(x)
    elif cell_type == "LTC":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(x)
    elif cell_type == "mmRNN":
        x = RNN(ODELSTM(64), return_sequences=False)(x)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(64), return_sequences=False)(x)
    elif cell_type == "CT-GRU":
        x = RNN(GRUODE(64), return_sequences=False)(x)
    elif cell_type == "CTRNN":
        x = RNN(CTRNNCell(64, num_unfolds=5, method='euler'), return_sequences=False)(x)
    elif cell_type == "NODE":
        x = RNN(NODE(units=64, hidden_dim=64, num_unfolds=5),return_sequences=False)(x)
    elif cell_type == "DeepState":
        x = DeepState(dim=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "S4":
        x = S4(d_model=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "Mamba":
        x = Mamba(d_model=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "ODEFormer":
        x = ODEformer(hidden_dim=64, num_heads=8, ff_dim=128)(x)
        x = Flatten()(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64, num_heads=16)(x)
    elif cell_type == 'ContiFormer':
        x = tf.keras.layers.Dense(64)(x)         # project to expected dim
        x = ContiFormer(dim=64, num_heads=8, ff_dim=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "PDE-Attention":
        x = PDEAttention(key_dim=16, num_heads=8, nt=5, dt=0.1, alpha=0.1)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "OT-Transformer":
        x = OTTransformer(key_dim=64, num_heads=16,ff_dim=32, num_steps=5)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=8)(x)
        x = Flatten()(x)
    elif cell_type == "NAC":
        x = NAC(d_model=64, num_heads=16, topk=32, return_sequences=False)(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Dense(32, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

# MODEL TYPES
model_types = [
    "LSTM", "GRU", "SDPA-Transformer",  #DT-models
    "NODE", "PhasedLSTM","mmRNN",         #F1
    "CT-GRU", "CT-RNN",'LTC','CfC', "FLUID", "NAC", #F2
    'DeepState', "S4",      #F3
    "Mamba", #F4
    "mTAN", "ODEFormer", "ContiFormer", 'OT-Transformer', 'PDE-Attention', #F5
]


# ---- Callbacks ----
def get_callbacks(model_name):
    return [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3,
                          min_lr=1e-10, mode='max'),
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# ---- K-Fold CV ----
k_folds = 5
results = {}

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_events, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X_events[train_idx], X_events[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

        # Build model
        model = build_model(cell_type, input_shape=(256,2), num_classes=10)
        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=30,callbacks=callbacks, verbose=0)

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        _, val_acc = model.evaluate(val_ds, verbose=0)

        fold_accuracies.append(val_acc * 100)

    # Store CV results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    results[cell_type] = {"fold_accuracies": fold_accuracies,
                          "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold Accuracies: {fold_accuracies}")
    print(f"{model_name} Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

# ---- Final summary ----
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: "
          f"Mean={data['mean']:.2f}%, Std={data['std']:.2f}%")
