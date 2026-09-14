"""Exercise real generation orchestration and PNG records with a CPU model double."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

from PIL import Image
from backend.preference.config import DEFAULTS, split_for_prompt
from backend.preference.generation import generate
from backend.preference.store import PreferenceStore


class PromptGenerationTests(unittest.TestCase):
    def test_variable_counts_negatives_seed_continuity_and_png_provenance(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            cfg = copy.deepcopy(DEFAULTS)
            cfg['dataset_dir'] = directory
            cfg['generation'].update(width=256, height=256, seed=100, prompts=[
                {'prompt': 'first\nline', 'negative_prompt': 'no blur', 'pairs': 2},
                {'prompt': 'second', 'negative_prompt': '', 'pairs': 3}])
            identity = {'reference_id': 'test-reference'}
            stack.enter_context(patch('backend.preference.generation.model_identity', return_value=identity))
            model = stack.enter_context(patch('backend.preference.generation.SDXLModel')).return_value
            model.identity = identity
            model.generate.side_effect = lambda *args: (Image.new('RGB', (256, 256)), {'test': True})
            for name, value in [('reset_peak_memory_stats', None), ('max_memory_allocated', 0),
                                ('max_memory_reserved', 0), ('get_device_name', 'CPU test')]:
                stack.enter_context(patch('backend.preference.generation.torch.cuda.' + name, return_value=value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            first = generate(cfg)
            second = generate(cfg)
            self.assertEqual((first['comparisons'], second['comparisons']), (5, 5))
            calls = model.generate.call_args_list
            self.assertEqual(len(calls), 20)
            self.assertEqual(sorted(call.args[3] for call in calls), list(range(100, 120)))
            self.assertEqual(Counter(call.args[0] for call in calls), {'first\nline': 8, 'second': 12})
            store = PreferenceStore(directory)
            rows = store.list_comparisons()
            self.assertEqual(len(rows), 10)
            for row in rows:
                expected_negative = 'no blur' if row['prompt'] == 'first\nline' else ''
                self.assertEqual(row['negative_prompt'], expected_negative)
                self.assertEqual(row['generation_settings']['negative_prompt'], expected_negative)
                group, split = split_for_prompt(row['prompt'], .2, 9796)
                self.assertEqual((row['group_id'], row['split']), (group, split))
                for image in row['images']:
                    with Image.open(store.root / image['path']) as img:
                        metadata = json.loads(img.info['preference_generation'])
                    self.assertEqual(metadata['negative_prompt'], expected_negative)
                    self.assertEqual(metadata['prompt'], row['prompt'])
                    self.assertEqual(metadata['seed'], image['seed'])
            self.assertEqual(store.counts()['rated'], 0)


if __name__ == '__main__':
    unittest.main()
