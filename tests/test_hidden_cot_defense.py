import unittest
from dataclasses import dataclass

from defended_target import HiddenCoTDefendedTarget, SAFE_FALLBACK_RESPONSE
from experiment_profiles import (
    PAIR_PAPER_PROFILE,
    PILOT_SMALL_PROFILE,
    PILOT_STRONG_DEPTH10_PROFILE,
)
from reasoning_target import TargetGeneration


@dataclass
class FakeJudgment:
    rationale: str
    label: str
    evidence: list[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class SequenceTarget:
    def __init__(self, generations):
        self.generations = generations
        self.calls = 0
        self.reasons = []

    def generate_with_reasoning(
        self, prompt, regeneration_count=0, regeneration_reason=""
    ):
        self.reasons.append(regeneration_reason)
        generation = self.generations[
            min(self.calls, len(self.generations) - 1)
        ]
        self.calls += 1
        return generation


class RecordingJudge:
    def __init__(self, mode, labels):
        self.mode = mode
        self.labels = labels
        self.calls = 0
        self.signals = []

    def classify(self, prompt, signal):
        self.signals.append(signal)
        label = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        return FakeJudgment(rationale=f"classified {label}", label=label)


def generation(response="response", reasoning="reasoning", finish_reason="stop"):
    return TargetGeneration(
        response=response,
        reasoning=reasoning,
        finish_reason=finish_reason,
        usage={"total_tokens": 10},
    )


class HiddenCoTDefenseTests(unittest.TestCase):
    def test_output_only_uses_response_and_releases(self):
        output = RecordingJudge("without_hidden_cot", ["safe_content"])
        defense = HiddenCoTDefendedTarget(
            SequenceTarget([generation()]), output, defense_mode="without_hidden_cot"
        )
        self.assertEqual(defense.get_response(["prompt"]), ["response"])
        self.assertEqual(output.signals, ["response"])
        self.assertIsNone(defense.last_defense_records[0]["cot_judge_label"])

    def test_additive_mode_judges_same_candidate_with_both_signals(self):
        output = RecordingJudge("without_hidden_cot", ["safe_content"])
        cot = RecordingJudge("with_hidden_cot", ["safe_plan"])
        defense = HiddenCoTDefendedTarget(
            SequenceTarget([generation()]),
            output,
            cot,
            defense_mode="with_hidden_cot",
        )
        self.assertEqual(defense.get_response(["prompt"]), ["response"])
        self.assertEqual(output.signals, ["response"])
        self.assertEqual(cot.signals, ["reasoning"])

    def test_output_rejection_regenerates_with_reason(self):
        target = SequenceTarget([generation("bad"), generation("safe")])
        output = RecordingJudge(
            "without_hidden_cot", ["unsafe_content", "safe_content"]
        )
        defense = HiddenCoTDefendedTarget(target, output, max_regenerations=3)
        self.assertEqual(defense.get_response(["prompt"]), ["safe"])
        record = defense.last_defense_records[0]
        self.assertEqual(record["regeneration_count"], 1)
        self.assertIn("unsafe_content", target.reasons[1])

    def test_cot_rejection_regenerates_even_when_output_is_safe(self):
        target = SequenceTarget([generation("safe1"), generation("safe2")])
        output = RecordingJudge(
            "without_hidden_cot", ["safe_content", "safe_content"]
        )
        cot = RecordingJudge(
            "with_hidden_cot", ["unsafe_plan", "defensive_decoy_plan"]
        )
        defense = HiddenCoTDefendedTarget(
            target, output, cot, defense_mode="with_hidden_cot"
        )
        self.assertEqual(defense.get_response(["prompt"]), ["safe2"])
        self.assertIn("unsafe_plan", target.reasons[1])

    def test_truncated_candidate_regenerates_without_judging(self):
        target = SequenceTarget(
            [generation("partial", finish_reason="length"), generation("safe")]
        )
        output = RecordingJudge("without_hidden_cot", ["safe_content"])
        defense = HiddenCoTDefendedTarget(target, output)
        self.assertEqual(defense.get_response(["prompt"]), ["safe"])
        self.assertEqual(output.signals, ["safe"])
        first = defense.last_defense_records[0]["attempts"][0]
        self.assertEqual(first["output_judge_label"], "incomplete_or_uncertain")

    def test_persistent_unsafe_output_releases_fallback_after_budget(self):
        target = SequenceTarget([generation("bad")])
        output = RecordingJudge("without_hidden_cot", ["unsafe_content"])
        defense = HiddenCoTDefendedTarget(target, output, max_regenerations=3)
        self.assertEqual(
            defense.get_response(["prompt"]), [SAFE_FALLBACK_RESPONSE]
        )
        record = defense.last_defense_records[0]
        self.assertEqual(record["regeneration_count"], 3)
        self.assertEqual(record["action"], "fallback_after_max_regenerations")
        self.assertEqual(record["output_judge_label"], "safe_content")
        self.assertEqual(target.calls, 4)

    def test_profiles_keep_paper_and_pilot_distinct(self):
        self.assertEqual(PAIR_PAPER_PROFILE["attack_model"], "mixtral")
        self.assertEqual(PAIR_PAPER_PROFILE["n_streams"], 30)
        self.assertFalse(PAIR_PAPER_PROFILE["hidden_cot_defense"])
        self.assertEqual(PILOT_SMALL_PROFILE["attack_model"], "qwen3-8b")
        self.assertTrue(PILOT_SMALL_PROFILE["hidden_cot_defense"])
        self.assertEqual(PILOT_STRONG_DEPTH10_PROFILE["n_streams"], 1)
        self.assertEqual(PILOT_STRONG_DEPTH10_PROFILE["n_iterations"], 10)


if __name__ == "__main__":
    unittest.main()
