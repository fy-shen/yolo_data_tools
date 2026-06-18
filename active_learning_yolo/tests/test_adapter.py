import unittest

import numpy as np
import torch

from active_learning_yolo.adapters.ultralytics import (
    extract_feature_maps,
    predict_with_object_features,
    results_to_predictions,
    sample_detection_features,
)


class FakeBoxes:
    xyxy = np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32)
    conf = np.array([0.7], dtype=np.float32)
    cls = np.array([2.0], dtype=np.float32)
    lvl_inds = np.array([1], dtype=np.int64)

    def __len__(self) -> int:
        return 1


class FakeResult:
    path = "/data/images/sample.jpg"
    boxes = FakeBoxes()
    orig_shape = (4, 4)


class FakeDetectionModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, source, stream=False, **kwargs):
        self.calls.append(("predict", source, kwargs))
        if kwargs.get("embed") is None:
            return [FakeResult() for _ in source]
        level0 = torch.ones((len(source), 2, 4, 4), dtype=torch.float32)
        level1 = torch.full((len(source), 2, 4, 4), 2.0, dtype=torch.float32)
        return [level0, level1]


class UltralyticsAdapterTest(unittest.TestCase):
    def test_result_conversion(self) -> None:
        predictions = results_to_predictions([FakeResult()])
        self.assertEqual(predictions[0].image_id, "sample")
        self.assertEqual(predictions[0].detections[0].class_id, 2)
        self.assertIsNone(predictions[0].detections[0].feature)

    def test_sample_detection_features_uses_level_indices(self) -> None:
        predictions = results_to_predictions([FakeResult()])
        level0 = torch.ones((1, 2, 4, 4), dtype=torch.float32)
        level1 = torch.full((1, 2, 4, 4), 2.0, dtype=torch.float32)
        sample_detection_features(
            predictions, [FakeResult()], [level0, level1], imgsz=4
        )
        np.testing.assert_allclose(
            predictions[0].detections[0].feature,
            np.array([2.0, 2.0], dtype=np.float32),
        )

    def test_multilevel_sampling_requires_level_indices(self) -> None:
        class BoxesWithoutLevel(FakeBoxes):
            lvl_inds = None

        class ResultWithoutLevel(FakeResult):
            boxes = BoxesWithoutLevel()

        predictions = results_to_predictions([ResultWithoutLevel()])
        maps = [
            torch.ones((1, 2, 4, 4), dtype=torch.float32),
            torch.ones((1, 2, 4, 4), dtype=torch.float32),
        ]
        with self.assertRaises(ValueError):
            sample_detection_features(
                predictions, [ResultWithoutLevel()], maps, imgsz=4
            )

    def test_extract_feature_maps_groups_layers(self) -> None:
        model = FakeDetectionModel()
        maps = extract_feature_maps(model, ["a.jpg"], embed_layers=[0, 10])
        self.assertEqual(len(maps), 2)
        self.assertEqual(tuple(maps[0].shape), (1, 2, 4, 4))
        self.assertEqual(model.calls[0][2]["embed"], [0, 10])

    def test_predict_with_object_features(self) -> None:
        model = FakeDetectionModel()
        predictions = predict_with_object_features(
            model, ["a.jpg"], embed_layers=[0, 10], imgsz=4
        )
        self.assertEqual(model.calls[0][2]["embed"], None)
        self.assertEqual(model.calls[1][2]["embed"], [0, 10])
        np.testing.assert_allclose(
            predictions[0].detections[0].feature,
            np.array([2.0, 2.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
