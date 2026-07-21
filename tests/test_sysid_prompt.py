from __future__ import annotations

import json
import sys
import types
import unittest

sys.modules.setdefault("openai", types.SimpleNamespace(AsyncOpenAI=object))

from scripts.llm_eval_sysid import SYSTEM_CONTEXT, build_user_prompt, parse_response


class ContextDisclosurePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.constants = {"g": 3.71, "L": [1.4], "m": [2.0], "damping": 0.05}
        self.history = [
            {"t": -0.1, "theta": [0.2], "omega": [0.1]},
            {"t": 0.0, "theta": [0.21], "omega": [0.09]},
        ]

    def test_context_arms_request_identical_output_schema(self) -> None:
        disclosed = build_user_prompt(
            1, [0.21], [0.09], "disclosed", self.constants, 1.0, history=self.history
        )
        hidden = build_user_prompt(
            1, [0.21], [0.09], "hidden", self.constants, 1.0, history=self.history
        )
        self.assertIn("inferred_constants", SYSTEM_CONTEXT)
        self.assertIn("inferred_constants", disclosed)
        self.assertIn("inferred_constants", hidden)
        self.assertIn("g (gravity", disclosed)
        self.assertIn("Physical constants: HIDDEN", hidden)

    def test_context_disclosed_response_parses_parameter_object(self) -> None:
        payload = json.dumps({
            "theta": [0.1], "omega": [0.2], "inferred_constants": self.constants,
        })
        theta, omega, inferred = parse_response(
            payload, 1, "disclosed", require_constants=True
        )
        self.assertEqual(theta, [0.1])
        self.assertEqual(omega, [0.2])
        self.assertEqual(inferred["g"], self.constants["g"])


if __name__ == "__main__":
    unittest.main()
