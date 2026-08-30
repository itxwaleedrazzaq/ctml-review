import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTMCell, GRUCell, RNN, GlobalAveragePooling1D
from tensorflow.keras.models import Model

#imported families architectures
from nns import *
from flops_counter import get_flops as _count_flops_impl
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


# SHARED HYPERPARAMETERS
stat_dir = 'statistics'
sequence_length = 1024                          # long sequence (>1000)
hidden_dim = ncp_units = ff_dim = 64            # model width (d_model/dim/units/embed_dim/key_dim/state_size)
num_heads = 16                                  # attention heads (default)
num_heads_light = 8                             # heads for the lighter attention modules (FLUID, NAC)
num_unfolds = 5                                 # ODE solver steps
d_state = 16                                    # S7/Jamba SSM state dimension
d_conv = 4                                      # S7/Jamba convolution width
num_steps = 5                                   # OT-Transformer iterations / PDE-Attention nt
batch_size = 1
num_runs = 5
np.random.seed(42)

ncp_wiring = AutoNCP(units=ncp_units, output_size=1)

# MODEL BUILDER
def build_model(cell_type, seq_len=sequence_length, hidden_dim=hidden_dim, num_heads=num_heads):
    inp = Input(shape=(seq_len, hidden_dim))
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
        x = FLUID(d_model=hidden_dim, num_heads=num_heads_light, num_layers=1, ff_dim=ff_dim, topk=8, max_len=seq_len)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NAC":
        x = NAC(d_model=hidden_dim, num_heads=num_heads_light, topk=8, return_sequences=False)(inp)
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

    return Model(inp, x)

def get_flops(model, batch_size=1, seq_len=sequence_length, hidden_dim=hidden_dim):
    """FLOPs (multiply&add pairs x2) counted from the frozen compute graph by
    ``flops_counter``. Returns None if the model cannot be traced (reported in
    the results table rather than aborting the benchmark)."""
    try:
        return _count_flops_impl(model, batch_size=batch_size, seq_len=seq_len,
                                 input_dim=hidden_dim)
    except Exception as e:  # noqa: BLE001 - report, don't abort the benchmark
        print(f"    FLOPs estimation failed for {model.name}: {type(e).__name__}: {e}")
        return None

def measure_runtime_and_memory(model, num_runs=10):
    dummy_input = np.random.randn(batch_size, sequence_length, hidden_dim).astype(np.float32)

    # Warm-up and parameter counting
    _ = model(dummy_input)
    params = model.count_params()

    # FLOPs (do it once after warm-up); None if the model cannot be traced.
    flops = get_flops(model, batch_size=batch_size, seq_len=sequence_length,
                      hidden_dim=hidden_dim)

    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0

    if gpu_available:
        tf.config.experimental.reset_memory_stats("GPU:0")

    runtimes = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        runtimes.append(end - start)

    runtimes = np.array(runtimes)
    mean_rt = runtimes.mean()
    std_rt = runtimes.std()
    throughput = 1.0 / mean_rt

    if gpu_available:
        print('Using GPU')
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        gpu_mem_usage = mem_info["peak"] / (1024 ** 2)
    else:
        import psutil
        print('Using CPU')
        process = psutil.Process(os.getpid())
        gpu_mem_usage = process.memory_info().rss / (1024 ** 2)

    del model

    return mean_rt, std_rt, throughput, gpu_mem_usage, params, flops


# MODEL TYPES
model_types = [
    "LSTM", "GRU", "SDPA-Transformer",  #DT-models
    'DeepState', "HiPPO", "S4", "S5",   #F1 (Linear dynamical systems)
    "CT-GRU", "CT-RNN", "PhasedLSTM",'LTC','CfC', "FLUID", "NAC", #F2
    "NODE", "NCDE", "NRDE", "NSDE", "mmRNN",      #F3 (Freely parameterized vector fields)
    "Mamba",  "Jamba", "RetNet", "S7",   #F4 (Selective SSMs)
    "mTAN", "ODEFormer", "ContiFormer", 'OT-Transformer', 'PDE-Attention', #F5
]


def benchmark(model_types=model_types, seq_len=sequence_length,
              num_runs=num_runs, hidden_dim=hidden_dim):
    """Run the runtime/memory/FLOPs benchmark for each model and return a
    pandas DataFrame. One failing model records a NaN row instead of aborting
    the whole benchmark."""
    results = []
    for cell_type in model_types:
        print(f"\nBenchmarking {cell_type}...")
        try:
            model = build_model(cell_type)
            mean_rt, std_rt, throughput, mem_usage, params, flops = \
                measure_runtime_and_memory(model, num_runs)
        except Exception as e:
            print(f"    Model '{cell_type}' FAILED: {type(e).__name__}: {e}")
            results.append({
                "Model": cell_type,
                "Sequence Length (n)": seq_len,
                "Hidden Dim (k)": hidden_dim,
                "Mean Runtime (s)": np.nan,
                "Std Dev (s)": np.nan,
                "Throughput (seq/s)": np.nan,
                "Peak Memory (MB)": np.nan,
                "Parameters": np.nan,
                "Total FLOPs": np.nan,
            })
            continue

        results.append({
            "Mean Runtime (s)": round(mean_rt, 4),
            "Std Dev (s)": round(std_rt, 4),
            "Throughput (seq/s)": round(throughput, 2),
            "Peak Memory (MB)": round(mem_usage, 2),
            "Total FLOPs": int(flops),
            "Parameters": params
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = benchmark()
    print("\n=== Runtime + Memory Benchmark Results ===")
    print(df.to_string(index=False))
