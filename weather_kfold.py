import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, Dense,RNN, LSTMCell, GRUCell, GlobalAveragePooling1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler

#imported families architectures
from nns import  *
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


base_model_name = 'Jenna_climate'
weights_dir = 'model_weights'

# Jena Climate is sampled every 10 mins. 
# 48 steps = 8 hours. We predict 24 steps = 6 hours ahead.
LOOKBACK = 48
HORIZON = 24
TARGET_COL = 'T (degC)'  # temperature

# SHARED HYPERPARAMETERS
hidden_dim = ff_dim = ncp_units = 64 # model width (d_model/dim/units/embed_dim/key_dim/state_size)
num_heads = 16                       # attention heads
num_unfolds = 5                      # ODE solver steps
d_state = 16                         # S7/Jamba SSM state dimension
d_conv = 4                           # S7/Jamba convolution width
num_steps = 5                        # OT-Transformer iterations / PDE-Attention nt
batch_size = 32
learning_rate = 0.001
epochs = 50
k_folds = 5
random_state = 42


def prepare_weather_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:15000]
    features = df.drop(columns=['Date Time'])
    cols = [c for c in features.columns if c != TARGET_COL] + [TARGET_COL]
    features = features[cols]
    n_features = features.shape[1]
    n = len(features)
    train_df = features[:int(n*0.6)]
    test_df = features[int(n*0.8):]
    scaler = StandardScaler()
    scaler.fit(train_df)
    train_s = scaler.transform(train_df)
    test_s = scaler.transform(test_df)

    def create_windows(data):
        X, y = [], []
        for i in range(len(data) - LOOKBACK - HORIZON):
            X.append(data[i:i+LOOKBACK, :])
            y.append(data[i+LOOKBACK:i+LOOKBACK+HORIZON, -1])  # target = last column
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_s)
    X_test, y_test = create_windows(test_s)

    return X_train, y_train, X_test, y_test, scaler, n_features

# Load data
X, y, X_test, y_test, scaler, n_feats = prepare_weather_data('data/weather.csv')

ncp_wiring = AutoNCP(units=ncp_units, output_size=HORIZON)

# Model builder
def build_model(cell_type, input_shape=(LOOKBACK, n_feats), num_outputs=HORIZON):
    inp = Input(shape=input_shape)
    if cell_type == "LSTM":
        x = RNN(LSTMCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRU":
        x = RNN(GRUCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "SDPA-Transformer":
        x = SPDATransformer(embed_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "CfC":
        x = CfC(ncp_wiring, return_sequences=False)(inp)
    elif cell_type == "LTC":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(inp)
    elif cell_type == "mmRNN":
        x = RNN(ODELSTM(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "CT-GRU":
        x = RNN(GRUODE(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "CT-RNN":
        x = RNN(CTRNNCell(hidden_dim, num_unfolds=num_unfolds, method='euler'), return_sequences=False)(inp)
    elif cell_type == "NODE":
        x = RNN(NODE(units=hidden_dim, hidden_dim=hidden_dim, num_unfolds=num_unfolds),return_sequences=False)(inp)
    elif cell_type == "DeepState":
        x = DeepState(dim=hidden_dim)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "S4":
        x = S4(d_model=hidden_dim)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "Mamba":
        x = Mamba(d_model=hidden_dim)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "ODEFormer":
        x = ODEformer(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=hidden_dim, num_heads=num_heads)(inp)
    elif cell_type == 'ContiFormer':
        x = tf.keras.layers.Dense(hidden_dim)(inp)         # project to expected dim
        x = ContiFormer(dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "PDE-Attention":
        x = PDEAttention(key_dim=hidden_dim, num_heads=num_heads, nt=num_steps, dt=0.1, alpha=0.1)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "OT-Transformer":
        x = OTTransformer(key_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim, num_steps=num_steps)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID":
        x = FLUID(d_model=hidden_dim, num_heads=num_heads, num_layers=1, ff_dim=ff_dim, topk=32, enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NAC":
        x = NAC(d_model=hidden_dim, num_heads=num_heads, topk=32, return_sequences=False)(inp)
    elif cell_type == "HiPPO":
        x = HiPPO(d_model=hidden_dim, state_size=hidden_dim)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "S5":
        x = S5(d_model=hidden_dim, state_size=hidden_dim, num_heads=num_heads)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NCDE":
        x = NCDE(d_model=hidden_dim, hidden_dim=hidden_dim, num_unfolds=num_unfolds)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NRDE":
        x = NRDE(d_model=hidden_dim, hidden_dim=hidden_dim, num_unfolds=num_unfolds)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NSDE":
        x = NSDE(d_model=hidden_dim, hidden_dim=hidden_dim, num_unfolds=num_unfolds)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "S7":
        x = S7(d_model=hidden_dim, d_state=d_state, d_conv=d_conv)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "Jamba":
        x = Jamba(d_model=hidden_dim, num_heads=num_heads, d_state=d_state, d_conv=d_conv)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "RetNet":
        x = RetNet(d_model=hidden_dim, num_heads=num_heads)(inp)
        x = GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Activation('relu')(x)
    out = Dense(num_outputs, activation='linear')(x) #predict all future steps at once
    return Model(inp, out)

# Callbacks
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# MODEL TYPES
model_types = [
    "LSTM", "GRU", "SDPA-Transformer",  #DT-models
    'DeepState', "HiPPO", "S4", "S5",   #F1 (Linear dynamical systems)
    "CT-GRU", "CT-RNN", "PhasedLSTM",'LTC','CfC', "FLUID", "NAC", #F2
    "NODE", "NCDE", "NRDE", "NSDE", "mmRNN",      #F3 (Freely parameterized vector fields)
    "Mamba",  "Jamba", "RetNet", "S7",   #F4 (Selective SSMs)
    "mTAN", "ODEFormer", "ContiFormer", 'OT-Transformer', 'PDE-Attention', #F5
]


# K-fold CV
results = {}

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_mse = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/{k_folds}")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))\
                                  .shuffle(buffer_size=5000).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(batch_size)

        model = build_model(cell_type)
        model.compile(optimizer=AdamW(learning_rate=learning_rate), loss="mse", metrics=['mae'])
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  callbacks=get_callbacks(f"{model_name}_fold{fold}"), verbose=0)

        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")

        mse, _ = model.evaluate(X_test, y_test, verbose=0)
        fold_mse.append(mse)

    # Store CV stats
    mean_acc = np.mean(fold_mse)
    std_acc = np.std(fold_mse)
    results[cell_type] = {"fold_mse": fold_mse, "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold MSE: {fold_mse}")
    print(f"{model_name} Mean MSE: {mean_acc:.4f}, Std: {std_acc:.4f}")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: Mean={data['mean']:.4f}, Std={data['std']:.4f}")
