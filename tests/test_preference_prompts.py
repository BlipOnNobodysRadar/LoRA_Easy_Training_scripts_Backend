import copy
import json
import tempfile
import unittest
from pathlib import Path

from backend.preference.config import DEFAULTS, load_config, prompt_entries, split_for_prompt


class PromptConfigTests(unittest.TestCase):
    def test_legacy_and_explicit_overrides_preserve_multiline_and_empty_negative(self):
        generation = {"prompts": ["first\nsecond line", {"prompt": "other", "negative_prompt": "", "pairs": 5}],
                      "negative_prompt": "shared", "pairs_per_prompt": 20}
        original = copy.deepcopy(generation)
        entries = prompt_entries(generation)
        self.assertEqual(entries, [{"prompt": "first\nsecond line", "negative_prompt": "shared", "pairs": 20},
                                   {"prompt": "other", "negative_prompt": "", "pairs": 5}])
        self.assertEqual(generation, original)
        self.assertEqual(prompt_entries({"prompts": entries}), entries)

    def test_invalid_rows_are_rejected_instead_of_silently_skipped(self):
        for row in ("  ", {"prompt": "p", "pairs": 0}, {"prompt": "p", "pairs": True},
                    {"prompt": "p", "pairs": 2.5}, {"prompt": "p", "negative_prompt": None},
                    {"prompt": "p", "pair": 5}, None):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Prompt 1"):
                prompt_entries({"prompts": [row]})

    def test_load_old_and_new_configs_without_writing_or_affecting_training_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.safetensors').touch()
            path = root / 'config.json'
            raw = copy.deepcopy(DEFAULTS)
            raw['model']['checkpoint'] = 'model.safetensors'
            raw['generation'].update(prompts=['first', 'second'], negative_prompt='old negative', pairs_per_prompt=20)
            path.write_text(json.dumps(raw))
            before = path.read_bytes()
            loaded = load_config(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(loaded['generation']['prompts'][1]['pairs'], 20)
            raw['generation'] = loaded['generation']
            raw['generation']['prompts'][1].update(pairs=5, negative_prompt='')
            path.write_text(json.dumps(raw))
            updated = load_config(path)
            self.assertEqual(updated['training'], loaded['training'])
            self.assertEqual(updated['generation']['prompts'][1]['negative_prompt'], '')
            self.assertEqual(sum(p['pairs'] for p in updated['generation']['prompts']), 25)

    def test_same_positive_stays_in_one_split_even_with_different_negatives(self):
        entries = prompt_entries({'prompts': [{'prompt': 'Same Prompt', 'negative_prompt': 'blur'},
                                              {'prompt': ' same  prompt ', 'negative_prompt': 'text'}]})
        self.assertEqual(*(split_for_prompt(entry['prompt'], .2, 9796) for entry in entries))


if __name__ == '__main__':
    unittest.main()
