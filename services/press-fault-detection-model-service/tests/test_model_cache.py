import unittest
import sys
import os
from unittest.mock import Mock
import copy
import threading

# Add the parent directory to the path to import the model_cache
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.model_cache import model_cache


class TestModelCache(unittest.TestCase):
    """
    Comprehensive unit tests for the model_cache dictionary.

    Testing framework: unittest (Python standard library)

    This test suite covers:
    - Initial state validation
    - State mutations and integrity
    - Thread safety considerations
    - Edge cases and error conditions
    - Memory and reference management
    - Integration with actual service usage patterns
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Store original state to restore after each test
        self.original_cache = copy.deepcopy(model_cache)

    def tearDown(self):
        """Clean up after each test method."""
        # Restore original state
        model_cache.clear()
        model_cache.update(self.original_cache)

    def test_initial_cache_state(self):
        """Test that model_cache initializes with expected default values."""
        self.assertIsInstance(model_cache, dict)
        self.assertEqual(len(model_cache), 3)
        self.assertIn("model", model_cache)
        self.assertIn("scaler", model_cache)
        self.assertIn("threshold", model_cache)

    def test_initial_values_are_none(self):
        """Test that all initial values are None."""
        self.assertIsNone(model_cache["model"])
        self.assertIsNone(model_cache["scaler"])
        self.assertIsNone(model_cache["threshold"])

    def test_cache_keys_immutable_structure(self):
        """Test that the expected keys exist and are accessible."""
        expected_keys = {"model", "scaler", "threshold"}
        self.assertEqual(set(model_cache.keys()), expected_keys)

    def test_model_assignment_and_retrieval(self):
        """Test assigning and retrieving model objects."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1, 0, 1])

        model_cache["model"] = mock_model

        self.assertIs(model_cache["model"], mock_model)
        self.assertEqual(model_cache["model"].predict(), [1, 0, 1])

    def test_scaler_assignment_and_retrieval(self):
        """Test assigning and retrieving scaler objects."""
        mock_scaler = Mock()
        mock_scaler.transform = Mock(return_value=[[0.5, 0.3], [0.8, 0.1]])

        model_cache["scaler"] = mock_scaler

        self.assertIs(model_cache["scaler"], mock_scaler)
        self.assertEqual(model_cache["scaler"].transform(), [[0.5, 0.3], [0.8, 0.1]])

    def test_threshold_assignment_and_retrieval(self):
        """Test assigning and retrieving threshold values."""
        test_threshold = 0.75

        model_cache["threshold"] = test_threshold

        self.assertEqual(model_cache["threshold"], test_threshold)
        self.assertIsInstance(model_cache["threshold"], float)

    def test_threshold_with_various_numeric_types(self):
        """Test threshold assignment with different numeric types."""
        # Test with integer
        model_cache["threshold"] = 1
        self.assertEqual(model_cache["threshold"], 1)

        # Test with float
        model_cache["threshold"] = 0.85
        self.assertEqual(model_cache["threshold"], 0.85)

        # Test with negative values (edge case)
        model_cache["threshold"] = -0.1
        self.assertEqual(model_cache["threshold"], -0.1)

        # Test with zero
        model_cache["threshold"] = 0
        self.assertEqual(model_cache["threshold"], 0)

        # Test with very small float
        model_cache["threshold"] = 1e-10
        self.assertEqual(model_cache["threshold"], 1e-10)

    def test_cache_complete_population_scenario(self):
        """Test populating all cache entries simultaneously - mirrors main.py usage."""
        mock_model = Mock()
        mock_scaler = Mock()
        test_threshold = 0.6

        # Simulate main.py loading behavior
        model_cache["model"] = mock_model
        model_cache["scaler"] = mock_scaler
        model_cache["threshold"] = test_threshold

        self.assertIs(model_cache["model"], mock_model)
        self.assertIs(model_cache["scaler"], mock_scaler)
        self.assertEqual(model_cache["threshold"], test_threshold)

    def test_cache_partial_population_scenarios(self):
        """Test scenarios where only some cache entries are populated."""
        # Only model populated
        mock_model = Mock()
        model_cache["model"] = mock_model

        self.assertIs(model_cache["model"], mock_model)
        self.assertIsNone(model_cache["scaler"])
        self.assertIsNone(model_cache["threshold"])

        # Add scaler
        mock_scaler = Mock()
        model_cache["scaler"] = mock_scaler

        self.assertIs(model_cache["model"], mock_model)
        self.assertIs(model_cache["scaler"], mock_scaler)
        self.assertIsNone(model_cache["threshold"])

    def test_cache_reset_to_none(self):
        """Test resetting cache entries back to None."""
        # Populate cache
        model_cache["model"] = Mock()
        model_cache["scaler"] = Mock()
        model_cache["threshold"] = 0.8

        # Reset to None
        model_cache["model"] = None
        model_cache["scaler"] = None
        model_cache["threshold"] = None

        self.assertIsNone(model_cache["model"])
        self.assertIsNone(model_cache["scaler"])
        self.assertIsNone(model_cache["threshold"])

    def test_cache_overwrite_existing_values(self):
        """Test overwriting existing cache values - important for model reloading."""
        # Initial assignment
        old_model = Mock()
        old_scaler = Mock()
        old_threshold = 0.5

        model_cache["model"] = old_model
        model_cache["scaler"] = old_scaler
        model_cache["threshold"] = old_threshold

        # Overwrite with new values (model reload scenario)
        new_model = Mock()
        new_scaler = Mock()
        new_threshold = 0.9

        model_cache["model"] = new_model
        model_cache["scaler"] = new_scaler
        model_cache["threshold"] = new_threshold

        # Verify new values
        self.assertIs(model_cache["model"], new_model)
        self.assertIs(model_cache["scaler"], new_scaler)
        self.assertEqual(model_cache["threshold"], new_threshold)

        # Verify old values are not referenced
        self.assertIsNot(model_cache["model"], old_model)
        self.assertIsNot(model_cache["scaler"], old_scaler)
        self.assertNotEqual(model_cache["threshold"], old_threshold)

    def test_cache_invalid_key_access(self):
        """Test behavior when accessing non-existent keys."""
        with self.assertRaises(KeyError):
            _ = model_cache["non_existent_key"]

    def test_cache_invalid_key_assignment(self):
        """Test assigning to non-standard keys."""
        # This should work as it's a regular dict, but might indicate misuse
        model_cache["new_key"] = "new_value"
        self.assertEqual(model_cache["new_key"], "new_value")

        # Clean up
        del model_cache["new_key"]

    def test_cache_memory_references(self):
        """Test that cache stores references correctly."""
        original_object = {"data": [1, 2, 3]}
        model_cache["model"] = original_object

        # Modify original object
        original_object["data"].append(4)

        # Cache should reflect the change (same reference)
        self.assertEqual(model_cache["model"]["data"], [1, 2, 3, 4])

    def test_cache_with_none_explicit_assignment(self):
        """Test explicit None assignment vs implicit None."""
        # Explicit None assignment
        model_cache["model"] = None
        self.assertIsNone(model_cache["model"])
        self.assertTrue("model" in model_cache)

    def test_cache_get_method_like_predict_service(self):
        """Test using dict.get() method as used in predict_service.py."""
        # Test with None values (initial state)
        self.assertIsNone(model_cache.get("model"))
        self.assertIsNone(model_cache.get("scaler"))
        self.assertIsNone(model_cache.get("threshold"))

        # Test with default values
        self.assertEqual(model_cache.get("model", "default"), "default")
        self.assertEqual(model_cache.get("nonexistent", "default"), "default")

        # Test after assignment (loaded state)
        mock_model = Mock()
        mock_scaler = Mock()
        test_threshold = 0.8

        model_cache["model"] = mock_model
        model_cache["scaler"] = mock_scaler
        model_cache["threshold"] = test_threshold

        self.assertIs(model_cache.get("model"), mock_model)
        self.assertIs(model_cache.get("scaler"), mock_scaler)
        self.assertEqual(model_cache.get("threshold"), test_threshold)

    def test_cache_clear_functionality_like_main_shutdown(self):
        """Test clearing the cache as done in main.py shutdown."""
        # Populate cache
        model_cache["model"] = Mock()
        model_cache["scaler"] = Mock()
        model_cache["threshold"] = 0.6

        # Clear cache (simulate shutdown)
        model_cache.clear()

        self.assertEqual(len(model_cache), 0)
        self.assertFalse("model" in model_cache)
        self.assertFalse("scaler" in model_cache)
        self.assertFalse("threshold" in model_cache)

        # Restore original structure for other tests
        model_cache.update({
            "model": None,
            "scaler": None,
            "threshold": None
        })

    def test_cache_update_method(self):
        """Test updating cache with dict.update()."""
        new_values = {
            "model": Mock(),
            "scaler": Mock(),
            "threshold": 0.95
        }

        model_cache.update(new_values)

        self.assertIs(model_cache["model"], new_values["model"])
        self.assertIs(model_cache["scaler"], new_values["scaler"])
        self.assertEqual(model_cache["threshold"], new_values["threshold"])

    def test_cache_thread_safety_basic(self):
        """Test basic thread safety with concurrent access."""
        results = []
        errors = []

        def worker_read():
            try:
                # Simulate predict_service.py reading
                model = model_cache.get("model")
                scaler = model_cache.get("scaler")
                threshold = model_cache.get("threshold")
                results.append((model, scaler, threshold))
            except Exception as e:
                errors.append(e)

        def worker_write():
            try:
                # Simulate main.py writing
                model_cache["model"] = Mock()
                model_cache["scaler"] = Mock()
                model_cache["threshold"] = 0.5
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=worker_read)
            t2 = threading.Thread(target=worker_write)
            threads.extend([t1, t2])

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")

    def test_cache_complex_objects_storage(self):
        """Test storing complex ML objects in cache."""
        # Simulate complex model object
        complex_model = {
            "weights": [[0.1, 0.2], [0.3, 0.4]],
            "bias": [0.1, 0.2],
            "metadata": {"version": "1.0", "trained_on": "2024-01-01"}
        }

        # Simulate complex scaler object
        complex_scaler = {
            "mean_": [1.0, 2.0, 3.0],
            "scale_": [0.5, 0.8, 1.2],
            "feature_names": ["feature1", "feature2", "feature3"]
        }

        model_cache["model"] = complex_model
        model_cache["scaler"] = complex_scaler
        model_cache["threshold"] = 0.75

        self.assertEqual(model_cache["model"]["weights"], [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(model_cache["model"]["metadata"]["version"], "1.0")
        self.assertEqual(model_cache["scaler"]["mean_"], [1.0, 2.0, 3.0])
        self.assertEqual(model_cache["scaler"]["feature_names"], ["feature1", "feature2", "feature3"])

    def test_cache_callable_model_objects(self):
        """Test storing callable ML model objects in cache."""
        def dummy_model_predict(x):
            return [1 if val > 0.5 else 0 for val in x]

        def dummy_scaler_transform(x):
            return [[val * 2 for val in row] for row in x]

        model_cache["model"] = dummy_model_predict
        model_cache["scaler"] = dummy_scaler_transform
        model_cache["threshold"] = 0.5

        # Test model functionality
        self.assertIsInstance(model_cache["model"], type(dummy_model_predict))
        self.assertEqual(model_cache["model"]([0.3, 0.7, 0.9]), [0, 1, 1])

        # Test scaler functionality
        self.assertEqual(
            model_cache["scaler"]([[0.1, 0.2], [0.3, 0.4]]),
            [[0.2, 0.4], [0.6, 0.8]]
        )

    def test_cache_edge_case_threshold_values(self):
        """Test edge cases for threshold values."""
        edge_cases = [
            float('inf'),  # Infinity
            float('-inf'), # Negative infinity
            0.0,           # Zero
            1.0,           # One
            1e-15,         # Very small positive
            -1e-15,        # Very small negative
            999999.999999  # Very large
        ]

        for threshold in edge_cases:
            with self.subTest(threshold=threshold):
                model_cache["threshold"] = threshold
                self.assertEqual(model_cache["threshold"], threshold)

    def test_cache_nan_threshold_handling(self):
        """Test handling of NaN values in threshold."""
        import math

        model_cache["threshold"] = float('nan')

        # NaN should be stored but not equal to itself
        self.assertTrue(math.isnan(model_cache["threshold"]))

    def test_cache_none_vs_missing_key_distinction(self):
        """Test distinction between None values and missing keys."""
        # Initial state - keys exist with None values
        self.assertTrue("model" in model_cache)
        self.assertTrue("scaler" in model_cache)
        self.assertTrue("threshold" in model_cache)
        self.assertIsNone(model_cache["model"])

        # Delete a key
        del model_cache["model"]
        self.assertFalse("model" in model_cache)

        with self.assertRaises(KeyError):
            _ = model_cache["model"]

        # But .get() should still work
        self.assertIsNone(model_cache.get("model"))

    def test_cache_bool_coercion_scenarios(self):
        """Test boolean coercion scenarios that might occur in conditionals."""
        # Empty/falsy values
        falsy_values = [None, 0, 0.0, False, "", []]

        for value in falsy_values:
            with self.subTest(value=value):
                model_cache["threshold"] = value
                if model_cache["threshold"]:
                    self.fail(f"Value {value} should be falsy")

        # Truthy values
        truthy_values = [1, 0.1, True, "string", [1], {"key": "value"}]

        for value in truthy_values:
            with self.subTest(value=value):
                model_cache["threshold"] = value
                if not model_cache["threshold"]:
                    self.fail(f"Value {value} should be truthy")

    def test_cache_memory_efficiency(self):
        """Test that cache doesn't unnecessarily duplicate objects."""
        large_object = {"data": list(range(10000))}

        model_cache["model"] = large_object

        # Should be the same object (same memory reference)
        self.assertIs(model_cache["model"], large_object)

        # Modifying original should affect cached version
        large_object["data"].append(10000)
        self.assertEqual(len(model_cache["model"]["data"]), 10001)

    def test_cache_serialization_compatibility(self):
        """Test compatibility with common serialization scenarios."""
        import pickle
        import json

        # Test with serializable objects
        serializable_model = {"weights": [1, 2, 3], "bias": 0.5}
        model_cache["model"] = serializable_model
        model_cache["threshold"] = 0.8

        # Should be able to pickle the values
        pickled_model = pickle.dumps(model_cache["model"])
        unpickled_model = pickle.loads(pickled_model)
        self.assertEqual(unpickled_model, serializable_model)

        # Should be able to JSON serialize threshold
        json_threshold = json.dumps(model_cache["threshold"])
        self.assertEqual(json.loads(json_threshold), 0.8)

    def test_cache_error_recovery(self):
        """Test cache behavior during error conditions."""
        # Simulate partial loading failure
        model_cache["model"] = Mock()

        try:
            # Simulate scaler loading failure
            raise ValueError("Scaler loading failed")
        except ValueError:
            # Cache should maintain partial state
            self.assertIsNotNone(model_cache["model"])
            self.assertIsNone(model_cache["scaler"])
            self.assertIsNone(model_cache["threshold"])

    def test_cache_state_validation_helpers(self):
        """Test helper methods for validating cache state."""
        def is_cache_fully_loaded():
            return all(model_cache.get(key) is not None
                      for key in ["model", "scaler", "threshold"])

        def is_cache_empty():
            return all(model_cache.get(key) is None
                      for key in ["model", "scaler", "threshold"])

        # Initially empty
        self.assertTrue(is_cache_empty())
        self.assertFalse(is_cache_fully_loaded())

        # Partially loaded
        model_cache["model"] = Mock()
        self.assertFalse(is_cache_empty())
        self.assertFalse(is_cache_fully_loaded())

        # Fully loaded
        model_cache["scaler"] = Mock()
        model_cache["threshold"] = 0.7
        self.assertFalse(is_cache_empty())
        self.assertTrue(is_cache_fully_loaded())


if __name__ == '__main__':
    unittest.main()