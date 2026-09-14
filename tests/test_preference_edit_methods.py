import copy
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from backend.preference.config import DEFAULTS, utc_now
from backend.preference.import_pairs import import_pairs
from backend.preference.methods import method_settings, signature_settings
from backend.preference.objectives import objective_loss, leco_inputs
from backend.preference.sdxl import Cancelled
from backend.preference.store import PreferenceStore
from backend.preference.training import eligible_records


class GaussianDenoiser:
    """Analytic Gaussian noise predictor; theta translates the clean-image mean."""
    def __init__(self):
        self.theta = torch.tensor(0., requires_grad=True)
        self.preference = self
        self.preference_weight = 1.
        self.multiplier = 1.

    def set_multiplier(self, value):
        self.multiplier = value

    @contextmanager
    def reference_mode(self):
        self.multiplier = 0.
        try:
            with torch.no_grad():
                yield
        finally:
            self.multiplier = self.preference_weight

    def predict(self, noisy, timestep, text, vector):
        return noisy - self.multiplier * self.theta


class EditObjectiveTests(unittest.TestCase):
    def test_leco_partial_sampling_can_stop_before_a_denoiser_forward(self):
        model = GaussianDenoiser()
        model.device, model.dtype = torch.device('cpu'), torch.float32
        model.encode_prompt = lambda *args: (torch.zeros(1,1,1), torch.zeros(1,1))
        model.predict = lambda *args: self.fail('No denoiser call should occur after cancellation')
        settings = {'objective':'leco','leco':{'target':'scene','positive':'concept','denoising_steps':2}}
        with self.assertRaises(Cancelled):
            leco_inputs(model,settings,{'width':256,'height':256},7,cancelled=lambda:True)
        self.assertEqual(model.multiplier, 1.)

    def test_leco_enhance_and_erase_use_opposite_concept_targets(self):
        class TextDenoiser(GaussianDenoiser):
            def predict(self, noisy, timestep, text, vector):
                return noisy * 0 + text + self.multiplier * self.theta
        settings = copy.deepcopy(DEFAULTS['training'])
        settings.update(objective='leco',leco={'target':'scene','positive':'concept','guidance_scale':2.})
        z = torch.zeros(1,1,1,1)
        embeddings = {name:(torch.full_like(z,value),z) for name,value in
                      [('positive',3.),('neutral',1.),('unconditional',2.)]}
        for action, target in [('enhance',3.),('erase',-1.)]:
            model = TextDenoiser()
            settings['leco']['action'] = action
            loss,_ = objective_loss(model,(z.clone(),torch.ones(1),z,z,embeddings),
                                    settings,{'feedback':{'strength':'normal'}})
            loss.backward()
            # One analytic gradient step of .5 reaches the requested target.
            self.assertAlmostEqual(-.5*model.theta.grad.item(), target, places=6)

    def test_both_addift_directions_learn_positive_clean_image_shift(self):
        settings = copy.deepcopy(DEFAULTS['training'])
        settings['objective'] = 'addift'
        row = {'feedback': {'strength': 'normal'}}
        # Equal diffusion noise: winner clean image is +1, rejected image is 0.
        inputs = (torch.tensor([1.3, .3]).reshape(2, 1, 1, 1), torch.ones(2),
                  torch.zeros(2, 1), torch.zeros(2, 1), torch.ones(2, 1, 1, 1) * .3)
        for step in (0, 1):
            model = GaussianDenoiser()
            loss, metrics = objective_loss(model, inputs, settings, row, step)
            loss.backward()
            self.assertLess(model.theta.grad.item(), 0)  # gradient descent increases clean-image mean
            with torch.no_grad():
                model.theta -= .5 * model.theta.grad
            self.assertAlmostEqual(model.theta.item(), 1., places=6)
            self.assertEqual(model.multiplier, -1. if step else 1.)
            with model.reference_mode():
                torch.testing.assert_close(model.predict(*inputs[:4]), inputs[0])

    def test_legacy_dpo_resume_settings_are_identical(self):
        old = copy.deepcopy(DEFAULTS['training'])
        expected = {k:v for k,v in old.items() if k not in ('max_steps','checkpoint_every','output_dir')}
        self.assertEqual(signature_settings(old, DEFAULTS['generation']), expected)
        old.update(objective='dpo', addift={'min_timestep': 401}, leco={'target':'unused'})
        self.assertEqual(signature_settings(old, DEFAULTS['generation']), expected)

    def test_invalid_or_empty_objective_cannot_start(self):
        for value in ({'objective':'typo'}, {'objective':'leco'},
                      {'objective':'addift','addift':{'min_timestep':900,'max_timestep':400}}):
            with self.assertRaises(ValueError):
                method_settings(value)

    def test_import_is_unrated_immutable_idempotent_and_split_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new('RGB', (256, 256), 'red').save(root/'before.png')
            Image.new('RGB', (256, 256), 'blue').save(root/'after.png')
            source_bytes = (root/'before.png').read_bytes()
            entries = [{'source':'before.png','target':'after.png','prompt':'a cup', 'aligned':True}]
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(entries))
            cfg = copy.deepcopy(DEFAULTS)
            cfg['dataset_dir'] = str(root/'data')
            cfg['generation']['validation_fraction'] = 0
            with patch('backend.preference.import_pairs.model_identity', return_value={'reference_id':'ref'}):
                self.assertEqual(import_pairs(cfg, manifest)['imported'], 1)
                cfg['generation']['validation_fraction'] = .999
                self.assertEqual(import_pairs(cfg, manifest)['duplicates_skipped'], 1)
            store = PreferenceStore(cfg['dataset_dir'])
            row = store.list_comparisons()[0]
            self.assertEqual(store.counts()['eligible'], 0)
            self.assertEqual(row['split'], 'train')
            self.assertEqual((root/'before.png').read_bytes(), source_bytes)
            store.put_feedback(row['id'], dict(preference='b', strength='normal',
                               quality_a=None, quality_b=None, reasons=[], updated_at=utc_now()))
            self.assertEqual(len(eligible_records(store,'ref',False,'addift')), 1)
            # Source changes do not affect the frozen imported image.
            (root/'before.png').write_bytes(b'changed externally')
            self.assertEqual(len(eligible_records(store,'ref',False,'addift')), 1)
            (store.root/row['images'][0]['path']).write_bytes(b'tampered copy')
            with self.assertRaises(ValueError):
                eligible_records(store,'ref',False,'addift')


if __name__ == '__main__':
    unittest.main()
