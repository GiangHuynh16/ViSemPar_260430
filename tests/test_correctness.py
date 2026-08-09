import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amr_utils import canonicalize_amr, validate_amr  # noqa: E402
from data_pipeline import make_MTUP3, make_MTUP4  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from inference import extract_amr_MTUP2, extract_amr_MTUP3, extract_amr_MTUP4  # noqa: E402
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

    def test_mtup3_and_mtup4_require_their_final_steps(self):
        amr = "(w / want-01 :ARG0 (b / boy))"
        mtup3 = make_MTUP3("câu", amr)
        mtup4 = make_MTUP4("câu", amr)
        self.assertIn("[STEP 3: AMR_WITH_VAR]", mtup3)
        self.assertIn("[STEP 4: AMR_WITH_VAR]", mtup4)
        self.assertEqual(extract_amr_MTUP3("[STEP 2: AMR_NO_VAR]\n(want-01)"), "")
        self.assertEqual(extract_amr_MTUP4("[STEP 3: AMR_NO_VAR]\n(want-01)"), "")
        response3 = "[STEP 3: AMR_WITH_VAR]\n" + amr
        response4 = "[STEP 4: AMR_WITH_VAR]\n" + amr
        self.assertEqual(extract_amr_MTUP3(response3), amr)
        self.assertEqual(extract_amr_MTUP4(response4), amr)

    def test_metrics_report_requested_fields(self):
        metrics = compute_metrics(["(a / thing)"], ["(b / thing)"])
        for key in ["f1", "valid_graph_rate", "concept", "relation", "attribute", "root_acc"]:
            self.assertIn(key, metrics)

    def test_smatch_is_reproducible(self):
        predictions = ["(a / thing :ARG0 (b / person))"]
        gold = ["(x / thing :ARG0 (y / person))"]
        first = compute_metrics(predictions, gold)["smatch"]
        second = compute_metrics(predictions, gold)["smatch"]
        self.assertEqual(first, second)

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
