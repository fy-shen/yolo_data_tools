import unittest

import numpy as np

from active_learning_yolo.data import AnnotationPool
from active_learning_yolo.ppal import (
    OBJECT_FEATURES,
    ClassQualityEMA,
    Detection,
    ImagePrediction,
    PPALSelector,
    binary_entropy,
    compute_class_weights,
)
from active_learning_yolo.ppal.distance import build_distance_matrix


class ActiveLearningCoreTest(unittest.TestCase):
    def test_entropy_is_largest_near_half(self) -> None:
        self.assertGreater(binary_entropy(0.5), binary_entropy(0.9))

    def test_hard_class_has_larger_weight(self) -> None:
        weights = compute_class_weights({0: 0.1, 1: 0.9})
        self.assertGreater(weights[0], weights[1])

    def test_quality_ema(self) -> None:
        quality = ClassQualityEMA(num_classes=2, momentum=0.0)
        values = quality.update([(0, 0.81, 1.0), (1, 1.0, 0.25)])
        self.assertGreater(values[0], values[1])

    def test_object_feature_distance(self) -> None:
        predictions = [
            ImagePrediction("a", [Detection(
                0, 0.9, [0, 0, 10, 10], np.array([1.0, 0.0])
            )]),
            ImagePrediction("b", [Detection(
                0, 0.8, [0, 0, 10, 10], np.array([0.9, 0.1])
            )]),
        ]
        matrix = build_distance_matrix(predictions, OBJECT_FEATURES)
        self.assertGreaterEqual(matrix[0, 1], 0.0)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_selector_reports_resolved_mode(self) -> None:
        predictions = []
        for index in range(8):
            predictions.append(ImagePrediction(
                image_id=index,
                detections=[Detection(
                    index % 2, 0.45 + index * 0.01, [0, 0, 20, 20],
                    np.array([np.cos(index), np.sin(index)], dtype=np.float32)
                )],
            ))
        result = PPALSelector(
            budget=2, candidate_multiplier=3,
            diversity_mode=OBJECT_FEATURES, seed=7
        ).select(predictions)
        self.assertEqual(len(result.selected_ids), 2)
        self.assertEqual(result.diversity_mode, OBJECT_FEATURES)

    def test_selector_requires_object_features(self) -> None:
        predictions = [
            ImagePrediction(i, [Detection(0, 0.5, [0, 0, 10, 10])])
            for i in range(3)
        ]
        with self.assertRaises(ValueError):
            PPALSelector(budget=1).select(predictions)

    def test_annotation_pool(self) -> None:
        pool = AnnotationPool(labeled=["a"], unlabeled=["b", "c"])
        pool.request_annotation(["b"])
        pool.mark_labeled(["b"])
        self.assertIn("b", pool.labeled)
        self.assertNotIn("b", pool.pending)


if __name__ == "__main__":
    unittest.main()
