import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Input, Dense, RNN, LSTMCell, GRUCell, GlobalAveragePooling1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

# imported families architectures
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


base_model_name = 'PhysioNet2012'
weights_dir = 'model_weights'
os.makedirs(weights_dir, exist_ok=True)

LEAK_COLS = ['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']
TARGET_COL = 'In-hospital_death'


def prepare_physionet_data(file_path):
    df = pd.read_csv(file_path, index_col=0, compression='gzip')

    assert TARGET_COL in df.columns, f"Missing target column '{TARGET_COL}'"

    y = df[TARGET_COL].astype(float).values
    X = df.drop(columns=LEAK_COLS + [TARGET_COL])
    print(f"Loaded {df.shape[0]} records, {X.shape[1]} features after dropping "
          f"{LEAK_COLS + [TARGET_COL]}")
    print(f"Positive rate: {y.mean():.3f}")
    return X.values, y


X, y = prepare_physionet_data('data/PhysionetChallenge2012-set-a.csv.gz')
n_feats = X.shape[1]

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X = X.reshape(-1, n_feats, 1)


# Model builder (binary classification -> single sigmoid output)
def build_model(cell_type, input_shape=(n_feats, 1), num_outputs=1):
    inp = Input(shape=input_shape)
    if cell_type == "LSTM":
        x = RNN(LSTMCell(64), return_sequences=False)(inp)
    elif cell_type == "GRU":
        x = RNN(GRUCell(64), return_sequences=False)(inp)
    elif cell_type == "SDPA-Transformer":
        x = SPDATransformer(embed_dim=64, num_heads=16, ff_dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "CfC":
        x = CfC(ncp_wiring, return_sequences=False)(inp)
    elif cell_type == "LTC":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(inp)
    elif cell_type == "mmRNN":
        x = RNN(ODELSTM(16), return_sequences=False)(inp)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(64), return_sequences=False)(inp)
    elif cell_type == "CT-GRU":
        x = RNN(GRUODE(64), return_sequences=False)(inp)
    elif cell_type == "CT-RNN":
        x = RNN(CTRNNCell(64, num_unfolds=5, method='euler'), return_sequences=False)(inp)
    elif cell_type == "NODE":
        x = RNN(NODE(units=64, hidden_dim=64, num_unfolds=5),return_sequences=False)(inp)
    elif cell_type == "DeepState":
        x = DeepState(dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "S4":
        x = S4(d_model=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "Mamba":
        x = Mamba(d_model=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "ODEFormer":
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64, num_heads=16)(inp)
    elif cell_type == 'ContiFormer':
        x = tf.keras.layers.Dense(64)(inp)         # project to expected dim
        x = ContiFormer(dim=64, num_heads=16, ff_dim=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "PDE-Attention":
        x = PDEAttention(key_dim=64, num_heads=16, nt=5, dt=0.1, alpha=0.1)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "OT-Transformer":
        x = OTTransformer(key_dim=64, num_heads=16, ff_dim=64, num_steps=5)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID":
        x = FLUID(d_model=64, num_heads=8, num_layers=1, ff_dim=64, topk=8)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NAC":
        x = NAC(d_model=64, num_heads=8, topk=8, return_sequences=False)(inp)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Activation('relu')(x)
    out = Dense(num_outputs, activation='sigmoid')(x)  # binary classification
    return Model(inp, out)


ncp_wiring = AutoNCP(units=64, output_size=1)  # single scalar output


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
    "NODE", "PhasedLSTM","mmRNN",         #F1
    "CT-GRU", "CT-RNN",'LTC','CfC', "FLUID", "NAC", #F2
    'DeepState', "S4",      #F3
    "Mamba", #F4
    "mTAN", "ODEFormer", "ContiFormer", 'OT-Transformer', 'PDE-Attention', #F5
]


# K-fold CV with AUC
k_folds = 5
results = {}

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_auc = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/{k_folds}")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        y_val_fold = y_val_fold.astype(np.float32)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)) \
                                  .shuffle(buffer_size=5000).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(32)

        model = build_model(cell_type)
        model.compile(optimizer=AdamW(learning_rate=0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=50,
                  callbacks=get_callbacks(f"{model_name}_fold{fold}"), verbose=0)

        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")

        # AUC needs predicted probabilities on the validation fold
        probs = model.predict(X_val_fold, verbose=0).ravel()
        if len(np.unique(y_val_fold)) < 2:
            print("    Skipping fold AUC: single class present in validation fold.")
            continue
        fold_auc.append(roc_auc_score(y_val_fold, probs))

    # Store CV stats
    if fold_auc:
        mean_auc = np.mean(fold_auc)
        std_auc = np.std(fold_auc)
    else:
        mean_auc = std_auc = float('nan')
    results[cell_type] = {"fold_auc": fold_auc, "mean": mean_auc, "std": std_auc}

    print(f"{model_name} Fold AUC: {fold_auc}")
    print(f"{model_name} Mean AUC: {mean_auc:.4f}, Std: {std_auc:.4f}")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: Mean={data['mean']:.4f}, Std={data['std']:.4f}")
