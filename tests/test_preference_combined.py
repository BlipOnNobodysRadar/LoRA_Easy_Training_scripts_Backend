import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from backend.preference.combined import add_state, export_combined, scaled_state


def linear_state(rank, alpha, convolution=False):
    generator = torch.Generator().manual_seed(rank)
    up = torch.randn(6, rank, generator=generator)
    down = torch.randn(rank, 4, generator=generator)
    if convolution:
        up = up[:, :, None, None]
        down = down[:, :, None, None].repeat(1, 1, 3, 3)
    return {'lora_up.weight': up, 'lora_down.weight': down, 'alpha': torch.tensor(float(alpha))}


def delta(state):
    rank = state['lora_down.weight'].shape[0]
    return (state['lora_up.weight'].flatten(1).double() @ state['lora_down.weight'].flatten(1).double()
            * float(state['alpha']) / rank)


class CombinedExportTests(unittest.TestCase):
    def test_unequal_ranks_and_alphas_match_sum_without_cross_terms(self):
        for convolution in (False, True):
            a, b = linear_state(2, 4, convolution), linear_state(3, 1, convolution)
            original = copy.deepcopy(a)
            combined = add_state(scaled_state(a, .5), scaled_state(b, -.75))
            torch.testing.assert_close(delta(combined), .5 * delta(a) - .75 * delta(b), atol=2e-6, rtol=2e-6)
            self.assertEqual(combined['lora_down.weight'].shape[0], 5)
            for key in a:
                self.assertTrue(torch.equal(a[key], original[key]))

    def test_export_preserves_text_conv_norms_provenance_and_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {f'{name}.{key}': value for name in ('lora_te1_linear', 'lora_unet_linear')
                    for key, value in linear_state(2, 4).items()}
            base.update({'lora_unet_conv.' + key: value for key, value in linear_state(2, 4, True).items()})
            base.update({'lora_unet_norm.w_norm': torch.ones(6), 'lora_unet_norm.b_norm': torch.arange(6).float()})
            pref = {'lora_unet_linear.' + key: value for key, value in linear_state(3, 1).items()}
            save_file(base, str(root/'base.safetensors'))
            save_file(pref, str(root/'pref.safetensors'), metadata={'preference_reference_id': 'ref',
                      'preference_synthetic_test': 'false'})
            config = {'model': {'base_loras': [{'path': str(root/'base.safetensors'), 'weight': .5}]},
                      'training': {'allow_synthetic': False}}
            before = (root/'base.safetensors').read_bytes()
            with patch('backend.preference.combined.model_identity', return_value={'reference_id': 'ref'}):
                result = export_combined(config, root/'pref.safetensors', root/'combined.safetensors')
                self.assertEqual(result['source_count'], 2)
                with self.assertRaisesRegex(ValueError, 'never overwritten'):
                    export_combined(config, root/'pref.safetensors', root/'base.safetensors')
            self.assertEqual((root/'base.safetensors').read_bytes(), before)
            combined = load_file(root/'combined.safetensors')
            self.assertEqual(set(combined), set(base))
            self.assertTrue(torch.equal(combined['lora_unet_norm.w_norm'], base['lora_unet_norm.w_norm']*.5))
            self.assertTrue(torch.equal(combined['lora_unet_norm.b_norm'], base['lora_unet_norm.b_norm']*.5))
            for name in ('lora_te1_linear', 'lora_unet_conv'):
                state = lambda weights: {k.split('.', 1)[1]: v for k, v in weights.items() if k.startswith(name+'.')}
                torch.testing.assert_close(delta(state(combined)), delta(state(base))*.5)
            with safe_open(root/'combined.safetensors', framework='pt') as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata['preference_export_type'], 'combined')
            self.assertEqual(len(json.loads(metadata['preference_sources'])), 2)

    def test_unsupported_algorithm_and_incompatible_shapes_fail_explicitly(self):
        for suffix in ('dora_scale', 'lora_mid.weight', 'hada_w1_a'):
            state = linear_state(2, 2)
            state[suffix] = torch.ones(2)
            with self.assertRaisesRegex(ValueError, 'supports plain'):
                scaled_state(state, 1)
        with self.assertRaisesRegex(ValueError, 'different target shapes'):
            add_state(scaled_state(linear_state(2, 2), 1), scaled_state(linear_state(2, 2, True), 1))

    def test_reference_mismatch_and_cancel_do_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({'lora_unet_x.' + k: v for k, v in linear_state(2, 2).items()},
                      str(root/'pref.safetensors'), metadata={'preference_reference_id': 'ref'})
            config = {'model': {'base_loras': []}, 'training': {'allow_synthetic': False}}
            with patch('backend.preference.combined.model_identity', return_value={'reference_id': 'different'}):
                with self.assertRaisesRegex(ValueError, 'different reference'):
                    export_combined(config, root/'pref.safetensors', root/'out.safetensors')
            with patch('backend.preference.combined.model_identity', return_value={'reference_id': 'ref'}):
                with self.assertRaises(InterruptedError):
                    export_combined(config, root/'pref.safetensors', root/'out.safetensors', lambda: True)
            self.assertFalse((root/'out.safetensors').exists())


if __name__ == '__main__':
    unittest.main()
