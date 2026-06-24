import unittest
from unittest import mock

import numpy as np
from active_learning_yolo.adapters.ultralytics import (
    iter_predict_with_object_features,
    predict_with_object_features,
    results_to_predictions,
)


class FakeBoxes:
    xyxy = np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32)
    conf = np.array([0.7], dtype=np.float32)
    cls = np.array([2.0], dtype=np.float32)

    def __len__(self) -> int:
        return 1


class FakeResult:
    path = "/data/images/sample.jpg"
    boxes = FakeBoxes()
    orig_shape = (4, 4)
    feats = np.array([[3.0, 4.0]], dtype=np.float32)


class UltralyticsAdapterTest(unittest.TestCase):
    def test_result_conversion(self) -> None:
        predictions = results_to_predictions([FakeResult()])
        self.assertEqual(predictions[0].image_id, "sample")
        self.assertEqual(predictions[0].detections[0].class_id, 2)
        self.assertIsNone(predictions[0].detections[0].feature)

    def test_predict_with_object_features_uses_native_result_feats(self) -> None:
        model = object()
        with mock.patch(
            "active_learning_yolo.adapters.ultralytics._predict_results_with_object_features",
            return_value=[FakeResult()],
        ) as predict_mock:
            predictions = predict_with_object_features(
                model, ["a.jpg"], image_ids=["image-1"], imgsz=4
            )

        predict_mock.assert_called_once()
        self.assertEqual(predictions[0].image_id, "image-1")
        np.testing.assert_allclose(
            predictions[0].detections[0].feature,
            np.array([3.0, 4.0], dtype=np.float32),
        )

    def test_iter_predict_with_object_features_streams_results(self) -> None:
        model = object()
        with mock.patch(
            "active_learning_yolo.adapters.ultralytics._iter_predict_results_with_object_features",
            return_value=iter([FakeResult()]),
        ) as predict_mock:
            predictions = list(iter_predict_with_object_features(
                model, ["a.jpg"], image_ids=["image-1"], imgsz=4
            ))

        predict_mock.assert_called_once()
        self.assertEqual(predictions[0].image_id, "image-1")
        np.testing.assert_allclose(
            predictions[0].detections[0].feature,
            np.array([3.0, 4.0], dtype=np.float32),
        )

    def test_predict_with_object_features_rejects_embed(self) -> None:
        with self.assertRaises(ValueError):
            predict_with_object_features(object(), ["a.jpg"], embed=[0, 10])


if __name__ == "__main__":
    unittest.main()
