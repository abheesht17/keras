import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.losses import MeanSquaredError
from keras.src.models import Model


class BatchNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_bn_basics(self):
        # vector case
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": False,
                "scale": False,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # image case, with regularizers
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "beta_regularizer": "l2",
                "gamma_regularizer": "l2",
            },
            call_kwargs={"training": True},
            input_shape=(2, 4, 4, 3),
            expected_output_shape=(2, 4, 4, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    @parameterized.product(
        axis=(-1, 1),
        input_shape=((5, 2, 3), (5, 3, 3, 2)),
        moving_mean_initializer=("zeros", "ones"),
        moving_variance_initializer=("zeros", "ones"),
    )
    def test_correctness(
        self,
        axis,
        input_shape,
        moving_mean_initializer,
        moving_variance_initializer,
    ):
        # Training
        layer = layers.BatchNormalization(
            axis=axis,
            momentum=0,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
        )
        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=input_shape)
        out = x
        for _ in range(3):
            out = layer(out, training=True)

        # Assert the normalization is correct.
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
        out = backend.convert_to_numpy(out)
        out = out - np.reshape(
            backend.convert_to_numpy(layer.beta), broadcast_shape
        )
        out = out / np.reshape(
            backend.convert_to_numpy(layer.gamma), broadcast_shape
        )

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        reduction_axes = tuple(reduction_axes)
        self.assertAllClose(np.mean(out, axis=reduction_axes), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=reduction_axes), 1.0, atol=1e-3)
        self.assertAllClose(layer.moving_mean, 0.0, atol=1e-3)
        self.assertAllClose(layer.moving_variance, 1.0, atol=1e-3)

        # Inference done before training shouldn't match.
        inference_out = layer(x, training=False)
        training_out = layer(x, training=True)
        self.assertNotAllClose(inference_out, training_out)

        # Since momentum is zero, inference after training should match.
        training_out = layer(x, training=True)
        inference_out = layer(x, training=False)
        self.assertAllClose(inference_out, training_out)

        # Masked result with no training should not differ
        x[:, 1, :] = 0.0
        unmasked_out = layer(x, training=False)
        masked = layers.Masking()(x)
        masked_out = layer(masked, training=False)
        self.assertAllClose(unmasked_out, masked_out)

        # Masked result should differ from unmasked result
        unmasked_out = layer(x, training=False)
        x[:, 1, :] = 0.0
        masked = layers.Masking()(x)
        masked_out = layer(masked, training=True)
        self.assertNotAllClose(unmasked_out, masked_out)

    @parameterized.product(
        synchronized=(
            (False, True) if backend.backend == "tensorflow" else (False,)
        ),
    )
    def test_input_fully_masked(self, synchronized):
        norm = layers.BatchNormalization(
            scale=False,
            center=False,
            synchronized=synchronized,
        )
        x = np.zeros((4, 5))
        mask = np.zeros((4,), dtype=np.float32)
        y = norm(x, mask=mask, training=True)
        self.assertAllClose(y, np.zeros_like(x, dtype=np.float32))

    @parameterized.product(run_eagerly=(True, False), mask_value=(0.0, 0.1, 1))
    @pytest.mark.requires_trainable_backend
    def test_bachnorm_ignore_masked_values(self, run_eagerly, mask_value):
        padded_data = np.array(
            [
                [
                    [1, 5],
                    [2, 5],
                    [mask_value, mask_value],
                    [mask_value, mask_value],
                ]
                for _ in range(10)
            ],
            dtype="float32",
        )

        inputs = layers.Input((None, 2))
        masked = layers.Masking(mask_value=mask_value)(inputs)
        normed = layers.BatchNormalization(momentum=0.0)(masked)
        model = Model(inputs, normed)
        loss = MeanSquaredError()
        model.compile(
            "rmsprop",
            loss=loss,
            run_eagerly=run_eagerly,
        )
        model.fit(x=padded_data, y=padded_data, batch_size=10, epochs=5)
        self.assertAllClose(model.layers[2].moving_mean.numpy(), [1.5, 5.0])
        self.assertAllClose(
            model.layers[2].moving_variance.numpy(), [0.25, 0.0]
        )

    def test_trainable_behavior(self):
        layer = layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=1e-7)
        layer.build((1, 4, 4, 3))
        layer.trainable = False
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)

        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(200, 4, 4, 3))

        out = layer(x, training=True)
        self.assertAllClose(out, x)

        layer.trainable = True
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)

        for _ in range(10):
            out = layer(x, training=True)

        out = backend.convert_to_numpy(out)
        out = out - np.reshape(
            backend.convert_to_numpy(layer.beta), (1, 1, 1, 3)
        )
        out = out / np.reshape(
            backend.convert_to_numpy(layer.gamma), (1, 1, 1, 3)
        )

        self.assertAllClose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-3)

    def test_large_value_within_autocast_scope(self):
        layer = layers.BatchNormalization()
        layer.build((1, 4, 4, 3))
        # Use 70000 to trigger overflow for float16
        large_value = ops.full(layer.moving_variance.shape, 70000)
        with backend.AutocastScope("float16"):
            layer.moving_variance.assign(large_value)
            self.assertAllClose(layer.moving_variance.value, large_value)

    def test_masked_broadcast_normalization(self):
        input_shape = (1, 2, 3, 4)
        mask_shape = (1, 2, 1)
        x = ops.ones(input_shape)
        mask = ops.ones(mask_shape)

        layer = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)

        y = layer(x, training=True, mask=mask)

        mean_y = ops.mean(y, axis=[0, 1, 2])

        self.assertAllClose(mean_y, ops.zeros((4,)), atol=1e-6)
        self.assertAllClose(y, ops.zeros_like(y), atol=1e-6)

        self.assertAllClose(layer.moving_mean, ops.ones((4,)), atol=1e-6)
        self.assertAllClose(layer.moving_variance, ops.zeros((4,)), atol=1e-6)

    @pytest.mark.requires_trainable_backend
    def test_renorm_basics(self):
        # Test basic renorm functionality
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "renorm": True,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            # moving_mean, moving_variance, moving_stddev, renorm_mean,
            # renorm_stddev
            expected_num_non_trainable_weights=5,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # Test renorm with clipping
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "renorm": True,
                "renorm_clipping": {"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
            },
            call_kwargs={"training": True},
            input_shape=(2, 4, 4, 3),
            expected_output_shape=(2, 4, 4, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=5,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_renorm_invalid_clipping_keys(self):
        with self.assertRaisesRegex(ValueError, "Received invalid keys"):
            layers.BatchNormalization(
                renorm=True, renorm_clipping={"invalid_key": 1.0}
            )

    def test_renorm_config_serialization(self):
        layer = layers.BatchNormalization(
            renorm=True,
            renorm_clipping={"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
            renorm_momentum=0.95,
        )
        config = layer.get_config()
        self.assertEqual(config["renorm"], True)
        self.assertEqual(
            config["renorm_clipping"], {"rmax": 3.0, "rmin": 0.3, "dmax": 5.0}
        )
        self.assertEqual(config["renorm_momentum"], 0.95)

        # Test that we can recreate the layer from config
        new_layer = layers.BatchNormalization.from_config(config)
        self.assertEqual(new_layer.renorm, True)
        self.assertEqual(
            new_layer.renorm_clipping, {"rmax": 3.0, "rmin": 0.3, "dmax": 5.0}
        )
        self.assertEqual(new_layer.renorm_momentum, 0.95)

    def test_renorm_variables_created(self):
        layer = layers.BatchNormalization(renorm=True)
        layer.build((None, 10))

        # Check renorm-specific variables are created
        self.assertTrue(hasattr(layer, "moving_stddev"))
        self.assertTrue(hasattr(layer, "renorm_mean"))
        self.assertTrue(hasattr(layer, "renorm_stddev"))

        # Check shapes
        self.assertEqual(layer.moving_stddev.shape, (10,))
        self.assertEqual(layer.renorm_mean.shape, (10,))
        self.assertEqual(layer.renorm_stddev.shape, (10,))

        # Non-renorm layer should not have these variables
        layer_no_renorm = layers.BatchNormalization(renorm=False)
        layer_no_renorm.build((None, 10))
        self.assertFalse(hasattr(layer_no_renorm, "moving_stddev"))
        self.assertFalse(hasattr(layer_no_renorm, "renorm_mean"))
        self.assertFalse(hasattr(layer_no_renorm, "renorm_stddev"))

    @parameterized.product(
        axis=(-1, 1),
        input_shape=((5, 2, 3), (5, 3, 3, 2)),
    )
    def test_renorm_correctness(self, axis, input_shape):
        # Training with renorm
        # Note: With renorm, output = (normalized * r + d) * gamma + beta
        # where r and d are correction factors. So the output won't have
        # exactly zero mean and unit variance like regular batch norm.
        layer = layers.BatchNormalization(
            axis=axis,
            momentum=0.0,
            renorm=True,
            renorm_momentum=0.0,
        )
        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=input_shape)

        # Call multiple times with the same input
        for _ in range(3):
            out = layer(x, training=True)

        # Verify output shape is correct
        self.assertEqual(out.shape, input_shape)

        # Verify output is valid (not NaN or Inf)
        out_np = backend.convert_to_numpy(out)
        self.assertFalse(np.any(np.isnan(out_np)))
        self.assertFalse(np.any(np.isinf(out_np)))

        # Verify moving statistics have been updated (since momentum=0)
        # Moving mean should be close to the input batch mean
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        reduction_axes = tuple(reduction_axes)
        x_np = np.array(x)
        expected_input_mean = np.mean(x_np, axis=reduction_axes)
        expected_input_var = np.var(x_np, axis=reduction_axes)
        self.assertAllClose(
            layer.moving_mean, expected_input_mean, atol=1e-3
        )
        self.assertAllClose(
            layer.moving_variance, expected_input_var, atol=1e-1
        )

        # Since momentum is zero, inference after training should use
        # the latest moving statistics
        training_out = layer(x, training=True)
        inference_out = layer(x, training=False)
        self.assertAllClose(inference_out, training_out, atol=1e-4)

    def test_renorm_clipping_effect(self):
        # Test that clipping is applied correctly
        layer = layers.BatchNormalization(
            renorm=True,
            renorm_clipping={"rmax": 1.5, "rmin": 0.5, "dmax": 0.5},
            momentum=0.0,
            renorm_momentum=0.99,  # High momentum to keep renorm stats stable
        )
        layer.build((None, 3))

        # Create data with high variance to trigger clipping
        x = np.array([[0.0, 50.0, -50.0], [100.0, -50.0, 50.0]], dtype="float32")

        # First call to initialize
        _ = layer(x, training=True)

        # The output should be valid (not NaN or Inf) due to clipping
        out = layer(x, training=True)
        out_np = backend.convert_to_numpy(out)
        self.assertFalse(np.any(np.isnan(out_np)))
        self.assertFalse(np.any(np.isinf(out_np)))

    def test_renorm_inference_same_as_regular_bn(self):
        # During inference, renorm should behave the same as regular BN
        # (using moving_mean and moving_variance)
        layer_renorm = layers.BatchNormalization(
            renorm=True,
            momentum=0.0,
        )
        layer_regular = layers.BatchNormalization(
            renorm=False,
            momentum=0.0,
        )

        x = np.random.normal(size=(4, 10)).astype("float32")

        # Build and set same moving stats
        layer_renorm.build((None, 10))
        layer_regular.build((None, 10))

        # Train both to update moving stats
        _ = layer_renorm(x, training=True)
        _ = layer_regular(x, training=True)

        # Copy moving stats from regular to renorm
        layer_renorm.moving_mean.assign(layer_regular.moving_mean)
        layer_renorm.moving_variance.assign(layer_regular.moving_variance)
        layer_renorm.gamma.assign(layer_regular.gamma)
        layer_renorm.beta.assign(layer_regular.beta)

        # Inference should be the same
        out_renorm = layer_renorm(x, training=False)
        out_regular = layer_regular(x, training=False)

        self.assertAllClose(out_renorm, out_regular, atol=1e-5)

    def test_renorm_stddev_initializer(self):
        # Test that moving_stddev and renorm_stddev are initialized as
        # sqrt of moving_variance_initializer
        layer = layers.BatchNormalization(
            renorm=True,
            moving_variance_initializer="ones",  # sqrt(1) = 1
        )
        layer.build((None, 10))

        # With variance initializer = ones, stddev should be 1.0
        self.assertAllClose(layer.moving_stddev, np.ones((10,)), atol=1e-6)
        self.assertAllClose(layer.renorm_stddev, np.ones((10,)), atol=1e-6)

        # Test with a different variance initializer (constant 4.0)
        # sqrt(4) = 2
        layer2 = layers.BatchNormalization(
            renorm=True,
            moving_variance_initializer=initializers.Constant(4.0),
        )
        layer2.build((None, 5))

        self.assertAllClose(
            layer2.moving_stddev, np.full((5,), 2.0), atol=1e-6
        )
        self.assertAllClose(
            layer2.renorm_stddev, np.full((5,), 2.0), atol=1e-6
        )
