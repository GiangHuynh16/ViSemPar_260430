import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amr_utils import canonicalize_amr, validate_amr  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from inference import extract_amr_MTUP2  # noqa: E402
from postprocess import postprocess_amr  # noqa: E402
from train import load_chat_dataset  # noqa: E402


class CharacterTokenizer:
    """Small deterministic tokenizer for testing assistant-label boundaries."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


class CorrectnessTests(unittest.TestCase):
    def test_source_multiword_concept_becomes_valid_penman(self):
        graph = canonicalize_amr("(b / bi kịch :domain (c / chỗ))")
        self.assertEqual(graph, "(b / bi_kịch :domain (c / chỗ))")
        self.assertTrue(validate_amr(graph))

    def test_postprocess_does_not_repair_invalid_graph(self):
        malformed = "(a / thing :ARG0 (b / person)"
        self.assertEqual(postprocess_amr(malformed), malformed)
        self.assertEqual(compute_metrics([malformed], ["(a / thing)"])["valid_graph_rate"], 0.0)

    def test_mtup2_requires_final_step(self):
        response = "[STEP 1: AMR_NO_VAR]\n(thing)"
        self.assertEqual(extract_amr_MTUP2(response), "")

    def test_metrics_report_requested_fields(self):
        metrics = compute_metrics(["(a / thing)"], ["(b / thing)"])
        for key in ["f1", "valid_graph_rate", "concept", "relation", "attribute", "root_acc"]:
            self.assertIn(key, metrics)

    def test_training_masks_only_before_assistant_response(self):
        example = (
            "<|im_start|>system\nS\n<|im_end|>\n"
            "<|im_start|>user\nCâu\n<|im_end|>\n"
            "<|im_start|>assistant\nAMR: (a / thing)\n<|im_end|>"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(example)
            handle.flush()
            dataset = load_chat_dataset(handle.name, CharacterTokenizer())
        self.assertEqual(len(dataset), 1)
        labels = dataset[0]["labels"]
        first_response = example.index("AMR:")
        self.assertTrue(all(label == -100 for label in labels[:first_response]))
        self.assertTrue(any(label != -100 for label in labels[first_response:]))


if __name__ == "__main__":
    unittest.main()
