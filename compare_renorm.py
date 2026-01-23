"""Compare Keras 2 (tf.keras) and Keras 3 batch renormalization implementations."""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create test input
x = np.random.normal(loc=5.0, scale=3.0, size=(4, 8)).astype("float32")

print("=" * 60)
print("Comparing Keras 2 (tf.keras) vs Keras 3 Batch Renormalization")
print("=" * 60)
print(f"\nInput shape: {x.shape}")
print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")

# ============== Keras 2 (tf.keras) ==============
print("\n" + "=" * 60)
print("Keras 2 (tf.keras) Implementation")
print("=" * 60)

import tensorflow as tf

tf_layer = tf.keras.layers.BatchNormalization(
    renorm=True,
    renorm_clipping={"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
    renorm_momentum=0.99,
    momentum=0.99,
)

# Build the layer
tf_layer.build((None, 8))

print("\nInitial state (Keras 2):")
print(f"  moving_mean: {tf_layer.moving_mean.numpy()}")
print(f"  moving_variance: {tf_layer.moving_variance.numpy()}")
print(f"  renorm_mean: {tf_layer.renorm_mean.numpy()}")
print(f"  renorm_stddev: {tf_layer.renorm_stddev.numpy()}")

# Training pass 1
tf_out1 = tf_layer(x, training=True)
print("\nAfter training pass 1 (Keras 2):")
print(f"  Output mean: {tf_out1.numpy().mean():.6f}")
print(f"  Output std: {tf_out1.numpy().std():.6f}")
print(f"  moving_mean: {tf_layer.moving_mean.numpy()}")
print(f"  moving_variance: {tf_layer.moving_variance.numpy()}")
print(f"  renorm_mean: {tf_layer.renorm_mean.numpy()}")
print(f"  renorm_stddev: {tf_layer.renorm_stddev.numpy()}")

# Training pass 2
tf_out2 = tf_layer(x, training=True)
print("\nAfter training pass 2 (Keras 2):")
print(f"  Output mean: {tf_out2.numpy().mean():.6f}")
print(f"  Output std: {tf_out2.numpy().std():.6f}")
print(f"  moving_mean: {tf_layer.moving_mean.numpy()}")
print(f"  moving_variance: {tf_layer.moving_variance.numpy()}")

# Inference pass
tf_out_inf = tf_layer(x, training=False)
print("\nInference output (Keras 2):")
print(f"  Output mean: {tf_out_inf.numpy().mean():.6f}")
print(f"  Output std: {tf_out_inf.numpy().std():.6f}")

# Store Keras 2 results
keras2_out1 = tf_out1.numpy()
keras2_out2 = tf_out2.numpy()
keras2_out_inf = tf_out_inf.numpy()
keras2_moving_mean = tf_layer.moving_mean.numpy().copy()
keras2_moving_var = tf_layer.moving_variance.numpy().copy()
keras2_renorm_mean = tf_layer.renorm_mean.numpy().copy()
keras2_renorm_stddev = tf_layer.renorm_stddev.numpy().copy()

# ============== Keras 3 ==============
print("\n" + "=" * 60)
print("Keras 3 Implementation")
print("=" * 60)

# Import Keras 3
import keras
from keras.src.layers.normalization.batch_normalization import BatchNormalization

k3_layer = BatchNormalization(
    renorm=True,
    renorm_clipping={"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
    renorm_momentum=0.99,
    momentum=0.99,
)

# Build the layer
k3_layer.build((None, 8))

print("\nInitial state (Keras 3):")
print(f"  moving_mean: {np.array(k3_layer.moving_mean)}")
print(f"  moving_variance: {np.array(k3_layer.moving_variance)}")
print(f"  renorm_mean: {np.array(k3_layer.renorm_mean)}")
print(f"  renorm_stddev: {np.array(k3_layer.renorm_stddev)}")

# Training pass 1
k3_out1 = k3_layer(x, training=True)
k3_out1_np = np.array(k3_out1)
print("\nAfter training pass 1 (Keras 3):")
print(f"  Output mean: {k3_out1_np.mean():.6f}")
print(f"  Output std: {k3_out1_np.std():.6f}")
print(f"  moving_mean: {np.array(k3_layer.moving_mean)}")
print(f"  moving_variance: {np.array(k3_layer.moving_variance)}")
print(f"  renorm_mean: {np.array(k3_layer.renorm_mean)}")
print(f"  renorm_stddev: {np.array(k3_layer.renorm_stddev)}")

# Training pass 2
k3_out2 = k3_layer(x, training=True)
k3_out2_np = np.array(k3_out2)
print("\nAfter training pass 2 (Keras 3):")
print(f"  Output mean: {k3_out2_np.mean():.6f}")
print(f"  Output std: {k3_out2_np.std():.6f}")
print(f"  moving_mean: {np.array(k3_layer.moving_mean)}")
print(f"  moving_variance: {np.array(k3_layer.moving_variance)}")

# Inference pass
k3_out_inf = k3_layer(x, training=False)
k3_out_inf_np = np.array(k3_out_inf)
print("\nInference output (Keras 3):")
print(f"  Output mean: {k3_out_inf_np.mean():.6f}")
print(f"  Output std: {k3_out_inf_np.std():.6f}")

# Store Keras 3 results
keras3_moving_mean = np.array(k3_layer.moving_mean)
keras3_moving_var = np.array(k3_layer.moving_variance)
keras3_renorm_mean = np.array(k3_layer.renorm_mean)
keras3_renorm_stddev = np.array(k3_layer.renorm_stddev)

# ============== Comparison ==============
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

def compare(name, keras2_val, keras3_val, atol=1e-5):
    diff = np.abs(keras2_val - keras3_val).max()
    match = diff < atol
    status = "✓ MATCH" if match else "✗ MISMATCH"
    print(f"{name}: {status} (max diff: {diff:.8f})")
    if not match:
        print(f"  Keras 2: {keras2_val}")
        print(f"  Keras 3: {keras3_val}")
    return match

all_match = True
all_match &= compare("Training output 1", keras2_out1, k3_out1_np)
all_match &= compare("Training output 2", keras2_out2, k3_out2_np)
all_match &= compare("Inference output", keras2_out_inf, k3_out_inf_np)
all_match &= compare("moving_mean", keras2_moving_mean, keras3_moving_mean)
all_match &= compare("moving_variance", keras2_moving_var, keras3_moving_var)
all_match &= compare("renorm_mean", keras2_renorm_mean, keras3_renorm_mean)
all_match &= compare("renorm_stddev", keras2_renorm_stddev, keras3_renorm_stddev)

print("\n" + "=" * 60)
if all_match:
    print("SUCCESS: All outputs match between Keras 2 and Keras 3!")
else:
    print("FAILURE: Some outputs differ between Keras 2 and Keras 3!")
print("=" * 60)
