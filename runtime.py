import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTMCell, GRUCell, RNN, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.python.framework import convert_to_constants

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


# CONFIG
stat_dir = 'statistics'
sequence_length = 1024   # long sequence (>1000)
hidden_dim = 64
batch_size = 1
num_runs = 5
num_heads = 4
np.random.seed(42)

ncp_wiring = AutoNCP(units=64,output_size=1)

# MODEL BUILDER
def build_model(cell_type, seq_len=sequence_length, hidden_dim=hidden_dim, num_heads=num_heads):
    inp = Input(shape=(seq_len, hidden_dim))

    if cell_type == "LSTM":
        x = RNN(LSTMCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRU":
        x = RNN(GRUCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "SDPA-Transformer":
        x = SPDATransformer(embed_dim=hidden_dim, num_heads=16, ff_dim=64)(inp)
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
        x = RNN(CTRNNCell(hidden_dim, num_unfolds=5, method='euler'), return_sequences=False)(inp)
    elif cell_type == "NODE":
        x = RNN(NODE(units=64, hidden_dim=64, num_unfolds=5),return_sequences=False)(inp)
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
        x = ODEformer(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=hidden_dim, num_heads=num_heads)(inp)
    elif cell_type == 'ContiFormer':
        x = tf.keras.layers.Dense(hidden_dim)(inp)         # project to expected dim
        x = ContiFormer(dim=hidden_dim, num_heads=num_heads, ff_dim=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "PDE-Attention":
        x = PDEAttention(key_dim=hidden_dim, num_heads=num_heads, nt=5, dt=0.1, alpha=0.1)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "OT-Transformer":
        x = OTTransformer(key_dim=hidden_dim, num_heads=num_heads,ff_dim=hidden_dim, num_steps=5)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID":
        x = FLUID(d_model=hidden_dim, num_heads=num_heads, num_layers=1, ff_dim=hidden_dim, topk=8, max_len=sequence_length)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "NAC":
        x = NAC(d_model=hidden_dim, num_heads=num_heads, topk=8, return_sequences=False)(inp)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    return Model(inp, x)

def get_flops(model, batch_size=1, seq_len=sequence_length, hidden_dim=hidden_dim):
    input_spec = tf.TensorSpec(shape=(batch_size, seq_len, hidden_dim), dtype=tf.float32)

    # Trace to a concrete function and freeze
    model_func = tf.function(model)
    concrete_func = model_func.get_concrete_function(input_spec)
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

    # Get graph from frozen function
    graph = frozen_func.graph
    # Get tensor names for input/output
    input_tensor_name = frozen_func.inputs[0].name
    output_tensor_name = frozen_func.outputs[0].name
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    dummy_input = tf.random.uniform(input_spec.shape, dtype=tf.float32)

    with tf.compat.v1.Session(graph=graph) as sess:
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()

        sess.run(output_tensor,
                 feed_dict={input_tensor: dummy_input},
                 options=run_options,
                 run_metadata=run_metadata)

        profiler = tf.compat.v1.profiler.profile(
            graph,
            run_meta=run_metadata,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        )
        total_flops = profiler.total_float_ops

    return total_flops

def measure_runtime_and_memory(model, num_runs=10):
    dummy_input = np.random.randn(batch_size, sequence_length, hidden_dim).astype(np.float32)

    # Warm-up and parameter counting
    _ = model(dummy_input)
    params = model.count_params()

    # FLOPs (do it once after warm-up)
    try:
        flops = get_flops(model, batch_size=batch_size, seq_len=sequence_length, hidden_dim=hidden_dim)
    except Exception as e:
        print(f"FLOPs estimation failed for {model.name}: {e}")
        flops = 0

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
    "NODE", "PhasedLSTM","mmRNN",         #F1
    "CT-GRU", "CT-RNN",'LTC','CfC', "FLUID", "NAC", #F2
    'DeepState', "S4",      #F3
    "Mamba", #F4
    "mTAN", "ODEFormer", "ContiFormer", 'OT-Transformer', 'PDE-Attention', #F5
]

# MAIN LOOP
results = []

for cell_type in model_types:
    print(f"\nBenchmarking {cell_type}...")
    model = build_model(cell_type)
    mean_rt, std_rt, throughput, mem_usage, params, flops = measure_runtime_and_memory(model, num_runs)

    results.append({
        "Model": cell_type,
        "Sequence Length (n)": sequence_length,
        "Hidden Dim (k)": hidden_dim,
        "Mean Runtime (s)": round(mean_rt, 4),
        "Std Dev (s)": round(std_rt, 4),
        "Throughput (seq/s)": round(throughput, 2),
        "Peak Memory (MB)": round(mem_usage, 2),
        "Parameters": params,
        "Total FLOPs": flops
    })

# RESULTS TABLE
df = pd.DataFrame(results)
print("\n=== Runtime + Memory Benchmark Results ===")
print(df.to_string(index=False))
