import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import api
from api import ImagePair
from matcher import HybridImageMatcher


class FakeProbabilities:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeInspector:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, **kwargs):
        del kwargs
        labels = list(self.scores)
        output = SimpleNamespace(
            probs=SimpleNamespace(
                data=FakeProbabilities([self.scores[label] for label in labels])
            ),
            names=dict(enumerate(labels)),
        )
        return [output]


class FailingInspector:
    def predict(self, **kwargs):
        del kwargs
        raise RuntimeError("inference unavailable")


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.images = ImagePair(
            QRCodeImage="reference",
            HouseImage="property",
            Remark="keep existing remark",
        )

    def inspect_with_scores(self, first_scores, second_scores):
        model_paths = [
            ("YOLO26n", Path(__file__)),
            ("YOLO11n", Path(__file__)),
        ]
        with (
            patch.object(api, "MODEL_PATHS", model_paths),
            patch.object(api, "get_inference_device", return_value="cpu"),
            patch.object(
                api,
                "get_model",
                side_effect=[
                    FakeInspector(first_scores),
                    FakeInspector(second_scores),
                ],
            ),
        ):
            return api.inspect_reference(Image.new("RGB", (200, 200)), self.images)

    def test_disagreeing_inspectors_use_confident_ensemble_decision(self):
        response = self.inspect_with_scores(
            {"accept+": 0.8, "gap between plate and wall": 0.2},
            {"accept+": 0.7, "gap between plate and wall": 0.3},
        )

        self.assertEqual(response["QRStatus"], 1)
        self.assertFalse(response["manual_review"])

    def test_low_confidence_margin_requires_review(self):
        scores = {"accept+": 0.51, "gap between plate and wall": 0.49}
        response = self.inspect_with_scores(scores, scores)

        self.assertIsNone(response["QRStatus"])
        self.assertTrue(response["manual_review"])
        self.assertIn("classification is uncertain", response["manual_review_reason"])

    def test_confident_agreement_preserves_automatic_approval(self):
        scores = {"accept+": 0.9, "gap between plate and wall": 0.1}
        response = self.inspect_with_scores(scores, scores)

        self.assertEqual(response["QRStatus"], 1)
        self.assertFalse(response["manual_review"])

    def test_moderate_confidence_predictions_are_automatic(self):
        scores = {"accept+": 0.53, "gap between plate and wall": 0.47}
        response = self.inspect_with_scores(scores, scores)

        self.assertEqual(response["QRStatus"], 1)
        self.assertFalse(response["manual_review"])

    def test_inspector_failure_requires_manual_review(self):
        with (
            patch.object(
                api,
                "MODEL_PATHS",
                [("YOLO26n", Path(__file__)), ("YOLO11n", Path(__file__))],
            ),
            patch.object(api, "get_inference_device", return_value="cpu"),
            patch.object(api, "get_model", return_value=FailingInspector()),
        ):
            response = api.inspect_reference(
                Image.new("RGB", (200, 200)), self.images
            )

        self.assertIsNone(response["QRStatus"])
        self.assertTrue(response["manual_review"])
        self.assertIn("failed to analyze", response["manual_review_reason"])
        self.assertEqual(response["Remark"], "keep existing remark")

    def test_missing_inspector_weights_require_manual_review(self):
        with patch.object(
            api,
            "MODEL_PATHS",
            [("missing", Path("/nonexistent/inspector.pt"))],
        ):
            response = api.inspect_reference(
                Image.new("RGB", (200, 200)), self.images
            )

        self.assertIsNone(response["QRStatus"])
        self.assertTrue(response["manual_review"])
        self.assertIn("weights are missing", response["manual_review_reason"])

    def test_weak_template_match_requires_review(self):
        matcher = HybridImageMatcher(device="cpu")
        reference = Image.new("RGB", (200, 200), "white")
        property_image = Image.new("RGB", (500, 500), "white")
        with (
            patch.object(matcher, "_find_qr_boxes", return_value=[[50, 50, 100, 100]]),
            patch.object(matcher, "_normalized_patch_sharpness", return_value=1000),
            patch.object(matcher, "_sharpness", return_value=100),
            patch.object(matcher, "_match_region", return_value=None),
            patch.object(
                matcher, "_template_search", return_value=[(0.44, [100, 100, 150, 150])]
            ),
            patch.object(matcher, "_region_detections", return_value=[]),
            patch.object(matcher, "_map_context_box", return_value=[100, 100, 150, 150]),
            patch.object(matcher, "_background_similarity", return_value=0.6),
        ):
            response = matcher.verify_presence(reference, property_image)

        self.assertTrue(response["needs_manual_review"])
        self.assertIn("visual match is too weak", response["manual_review_reason"])

    def test_blurry_property_qr_requires_review(self):
        matcher = HybridImageMatcher(device="cpu")
        reference = Image.new("RGB", (200, 200), "white")
        property_image = Image.new("RGB", (500, 500), "white")
        with (
            patch.object(matcher, "_find_qr_boxes", return_value=[[50, 50, 100, 100]]),
            patch.object(
                matcher, "_normalized_patch_sharpness", side_effect=[1000, 100]
            ),
            patch.object(matcher, "_sharpness", return_value=100),
            patch.object(matcher, "_match_region", return_value=None),
            patch.object(
                matcher, "_template_search", return_value=[(0.5, [100, 100, 150, 150])]
            ),
            patch.object(matcher, "_map_context_box", return_value=[100, 100, 150, 150]),
            patch.object(matcher, "_background_similarity", return_value=0.6),
        ):
            response = matcher.verify_presence(reference, property_image)

        self.assertTrue(response["needs_manual_review"])
        self.assertIn("property image is too blurry", response["manual_review_reason"])


if __name__ == "__main__":
    unittest.main()
