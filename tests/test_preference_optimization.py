import copy
import io
import unittest

from backend.preference.config import DEFAULTS
from backend.preference.methods import signature_settings
from backend.preference.optimization import optimization_settings, build_optimization


class OptimizationTests(unittest.TestCase):
    def test_legacy_signature_and_defaults_are_unchanged(self):
        settings = copy.deepcopy(DEFAULTS['training'])
        before = copy.deepcopy(settings)
        spec = optimization_settings(settings)
        self.assertEqual(settings, before)
        self.assertEqual(spec['optimizer'], {'type': 'AdamW', 'args': {'weight_decay': 0.}})
        self.assertEqual(spec['lr_schedule'], {'type': 'constant'})
        self.assertEqual(spec['max_grad_norm'], 1.)
        expected = {k:v for k,v in settings.items() if k not in
                    ('max_steps','checkpoint_every','output_dir','objective','addift','leco')}
        self.assertEqual(signature_settings(settings, DEFAULTS['generation']), expected)

    def test_horizon_is_fixed_for_schedules_and_auto_momentum(self):
        for extra in ({'lr_schedule': {'type':'rawr'}},
                      {'optimizer': {'type':'SimplifiedAdEMAMixExM','args':{'beta1_warmup':'total_steps'}}}):
            settings = {**copy.deepcopy(DEFAULTS['training']), **extra, 'max_steps':100}
            spec = optimization_settings(settings)
            self.assertEqual(spec['horizon_steps'],100)
            first = signature_settings(settings, DEFAULTS['generation'])
            settings['max_steps']=101
            self.assertNotEqual(first,signature_settings(settings,DEFAULTS['generation']))
        self.assertEqual(spec['optimizer']['args']['beta1_warmup'],100)

    def test_invalid_options_fail_before_gpu_initialization(self):
        invalid = [
            {'optimizer': 'AdamW'}, {'optimizer': {'type':'eval'}},
            {'optimizer': {'type':'AdamW','args': {'typo':1}}},
            {'optimizer': {'type':'AdamW','args': {'betas':[.9,1.]}}},
            {'optimizer': {'type':'SimplifiedAdEMAMixExM','args': {'beta3':100}}},
            {'optimizer': {'type':'SimplifiedAdEMAMixExM','args': {'torch_compile':'False'}}},
            {'optimizer': {'type':'SimplifiedAdEMAMixExM','args': {'beta1_warmup':101}}},
            {'optimizer': {'type':'SimplifiedAdEMAMixExM','args': {'state_storage_dtype':'bad'}}},
            {'lr_schedule': {'type':'rawr','warmup_ratio':1.}},
            {'lr_schedule': {'type':'rawr','min_lr':1.}},
            {'lr_schedule': {'type':'rawr','d':1.}},
            {'lr_schedule': {'type':'constant','warmup_ratio':.1}},
            {'max_grad_norm':True}, {'learning_rate':float('nan')},
        ]
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                optimization_settings({**DEFAULTS['training'],'max_steps':100,**extra})

    def test_native_adamw_rawr_resume_preserves_actual_updates(self):
        import torch
        cfg={'learning_rate':.0004,'max_steps':8,
             'lr_schedule':{'type':'rawr','warmup_ratio':.125}}
        def setup():
            p=torch.nn.Parameter(torch.tensor([.2,-.3]))
            opt,sched,_=build_optimization([p],cfg)
            return p,opt,sched
        def advance(p,opt,sched,start,end):
            rates=[]
            for i in range(start,end):
                p.grad=torch.tensor([.02+i*.001,-.01])
                rates.append(opt.param_groups[0]['lr'])
                opt.step(); sched.step()
            return rates
        p,opt,sched=setup(); expected_rates=advance(p,opt,sched,0,8); expected=p.detach().clone()
        p,opt,sched=setup(); rates=advance(p,opt,sched,0,3)
        buf=io.BytesIO()
        torch.save({'p':p.detach(),'optimizer':opt.state_dict(),'lr_scheduler':sched.state_dict()},buf)
        buf.seek(0); saved=torch.load(buf,weights_only=True)
        p,opt,sched=setup(); p.data.copy_(saved['p'])
        from backend.preference.training import restore_optimization
        with self.assertRaisesRegex(ValueError, 'missing its learning-rate scheduler'):
            restore_optimization(opt,sched,{'optimizer':saved['optimizer']})
        restore_optimization(opt,sched,saved)
        rates+=advance(p,opt,sched,3,8)
        self.assertEqual(rates,expected_rates)
        self.assertTrue(torch.equal(p,expected))
        self.assertEqual(rates[0],1e-6)
        self.assertEqual(rates[1],.0004)
        self.assertGreater(rates[1],rates[-1])


if __name__=='__main__':
    unittest.main()
