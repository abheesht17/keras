"""Compare Keras 2 (tf.keras) and Keras 3 batch renormalization implementations.

This script tests with:
1. Custom initial values for moving statistics
2. Multiple different batches during training
3. Various configurations (with/without clipping, different momentum values)
"""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create multiple different batches (10 batches with varying statistics)
batch1 = np.random.normal(loc=5.0, scale=3.0, size=(4, 8)).astype("float32")
batch2 = np.random.normal(loc=-2.0, scale=5.0, size=(4, 8)).astype("float32")
batch3 = np.random.normal(loc=10.0, scale=1.0, size=(4, 8)).astype("float32")
batch4 = np.random.normal(loc=0.0, scale=10.0, size=(4, 8)).astype("float32")
batch5 = np.random.normal(loc=-5.0, scale=2.0, size=(4, 8)).astype("float32")
batch6 = np.random.normal(loc=15.0, scale=0.5, size=(4, 8)).astype("float32")
batch7 = np.random.normal(loc=1.0, scale=8.0, size=(4, 8)).astype("float32")
batch8 = np.random.normal(loc=-10.0, scale=4.0, size=(4, 8)).astype("float32")
batch9 = np.random.normal(loc=3.0, scale=6.0, size=(4, 8)).astype("float32")
batch10 = np.random.normal(loc=0.0, scale=1.0, size=(4, 8)).astype("float32")

# Custom initial values (non-default)
init_moving_mean = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5], dtype="float32")
init_moving_variance = np.array([2.0, 3.0, 1.5, 2.5, 4.0, 0.5, 1.0, 3.5], dtype="float32")
init_moving_stddev = np.sqrt(init_moving_variance)
init_gamma = np.array([1.2, 0.8, 1.0, 1.5, 0.9, 1.1, 1.3, 0.7], dtype="float32")
init_beta = np.array([0.1, -0.1, 0.2, -0.2, 0.0, 0.3, -0.3, 0.15], dtype="float32")

import tensorflow as tf
import keras
from keras.src.layers.normalization.batch_normalization import BatchNormalization


def compare(name, keras2_val, keras3_val, atol=1e-5):
    diff = np.abs(keras2_val - keras3_val).max()
    match = diff < atol
    status = "✓ MATCH" if match else "✗ MISMATCH"
    print(f"  {name}: {status} (max diff: {diff:.8f})")
    if not match:
        print(f"    Keras 2: {keras2_val.flatten()[:4]}...")
        print(f"    Keras 3: {keras3_val.flatten()[:4]}...")
    return match


def run_comparison(
    test_name,
    renorm_clipping,
    momentum,
    renorm_momentum,
    use_custom_init=True,
):
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)
    print(f"  renorm_clipping: {renorm_clipping}")
    print(f"  momentum: {momentum}, renorm_momentum: {renorm_momentum}")
    print(f"  use_custom_init: {use_custom_init}")

    # ============== Keras 2 (tf.keras) ==============
    tf_layer = tf.keras.layers.BatchNormalization(
        renorm=True,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        momentum=momentum,
    )
    tf_layer.build((None, 8))

    # Set custom initial values
    if use_custom_init:
        tf_layer.moving_mean.assign(init_moving_mean)
        tf_layer.moving_variance.assign(init_moving_variance)
        tf_layer.moving_stddev.assign(init_moving_stddev)
        tf_layer.renorm_mean.assign(init_moving_mean)
        tf_layer.renorm_stddev.assign(init_moving_stddev)
        tf_layer.gamma.assign(init_gamma)
        tf_layer.beta.assign(init_beta)

    # ============== Keras 3 ==============
    k3_layer = BatchNormalization(
        renorm=True,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        momentum=momentum,
    )
    k3_layer.build((None, 8))

    # Set custom initial values
    if use_custom_init:
        k3_layer.moving_mean.assign(init_moving_mean)
        k3_layer.moving_variance.assign(init_moving_variance)
        k3_layer.moving_stddev.assign(init_moving_stddev)
        k3_layer.renorm_mean.assign(init_moving_mean)
        k3_layer.renorm_stddev.assign(init_moving_stddev)
        k3_layer.gamma.assign(init_gamma)
        k3_layer.beta.assign(init_beta)

    all_match = True

    # Training with multiple batches
    batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10]
    for i, batch in enumerate(batches):
        tf_out = tf_layer(batch, training=True)
        k3_out = k3_layer(batch, training=True)

        print(f"\n  After batch {i+1} (mean={batch.mean():.2f}, std={batch.std():.2f}):")
        all_match &= compare(f"output", np.asarray(tf_out), np.asarray(k3_out))
        all_match &= compare(
            f"moving_mean", np.asarray(tf_layer.moving_mean), np.asarray(k3_layer.moving_mean)
        )
        all_match &= compare(
            f"moving_var",
            np.asarray(tf_layer.moving_variance),
            np.asarray(k3_layer.moving_variance),
        )
        all_match &= compare(
            f"renorm_mean", np.asarray(tf_layer.renorm_mean), np.asarray(k3_layer.renorm_mean)
        )
        all_match &= compare(
            f"renorm_stddev",
            np.asarray(tf_layer.renorm_stddev),
            np.asarray(k3_layer.renorm_stddev),
        )

    # Inference pass
    print("\n  Inference mode:")
    for i, batch in enumerate(batches[:2]):
        tf_out_inf = tf_layer(batch, training=False)
        k3_out_inf = k3_layer(batch, training=False)
        all_match &= compare(
            f"inference batch {i+1}", np.asarray(tf_out_inf), np.asarray(k3_out_inf)
        )

    return all_match


# Run multiple test configurations
print("=" * 70)
print("Comparing Keras 2 (tf.keras) vs Keras 3 Batch Renormalization")
print("=" * 70)

all_tests_pass = True

# Test 1: With clipping, custom init
all_tests_pass &= run_comparison(
    test_name="With clipping, custom init",
    renorm_clipping={"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
    momentum=0.99,
    renorm_momentum=0.99,
    use_custom_init=True,
)

# Test 2: No clipping, custom init
all_tests_pass &= run_comparison(
    test_name="No clipping, custom init",
    renorm_clipping={},
    momentum=0.99,
    renorm_momentum=0.99,
    use_custom_init=True,
)

# Test 3: Different momentum values
all_tests_pass &= run_comparison(
    test_name="Low momentum (fast update)",
    renorm_clipping={"rmax": 2.0, "dmax": 3.0},
    momentum=0.5,
    renorm_momentum=0.8,
    use_custom_init=True,
)

# Test 4: Zero momentum (immediate update)
all_tests_pass &= run_comparison(
    test_name="Zero momentum",
    renorm_clipping={},
    momentum=0.0,
    renorm_momentum=0.0,
    use_custom_init=True,
)

# Test 5: Default init (zeros/ones)
all_tests_pass &= run_comparison(
    test_name="Default init (zeros/ones)",
    renorm_clipping={"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
    momentum=0.99,
    renorm_momentum=0.99,
    use_custom_init=False,
)

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
if all_tests_pass:
    print("✓ SUCCESS: All tests passed! Keras 2 and Keras 3 match perfectly.")
else:
    print("✗ FAILURE: Some tests failed! There are differences between implementations.")
print("=" * 70)
