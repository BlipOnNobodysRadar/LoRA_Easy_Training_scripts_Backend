import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from backend.preference.config import prompt_group, utc_now
from backend.preference.store import PreferenceStore
from backend.preference.training import eligible_records


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PreferenceStore(self.temp.name)
        images=[]
        for side in ('a','b'):
            path=self.store.root/(side+'.png')
            Image.new('RGB',(256,256),(123,80,22)).save(path)
            images.append(dict(id=side,path=path.name,seed=1,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.row=dict(id='one',created_at=utc_now(),session_id='s',group_id=prompt_group('a prompt'),
            split='train',synthetic=False,prompt='a prompt',negative_prompt='',model={'reference_id':'ref'},
            generation_settings={'width':256,'height':256},images=images)
        self.feedback=dict(preference='a',strength='slight',quality_a='bad',quality_b='bad',
                           reasons=[],updated_at=utc_now())

    def tearDown(self):
        self.temp.cleanup()

    def add(self, row=None):
        row=row or self.row
        self.store.add_comparison(row)
        self.store.put_feedback(row['id'],self.feedback)

    def test_real_pair_and_image_tampering(self):
        self.add()
        self.assertEqual(len(eligible_records(self.store,'ref',False)),1)
        (self.store.root/'a.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            eligible_records(self.store,'ref',False)

    def test_reference_mismatch_rejected(self):
        self.add()
        with self.assertRaisesRegex(ValueError,'different reference'):
            eligible_records(self.store,'other',False)

    def test_synthetic_labels_require_explicit_test_mode(self):
        self.row['synthetic']=True
        self.add()
        with self.assertRaisesRegex(ValueError,'No eligible'):
            eligible_records(self.store,'ref',False)
        self.assertEqual(len(eligible_records(self.store,'ref',True)),1)

    def test_no_prompt_can_leak_into_validation(self):
        self.add()
        other=copy.deepcopy(self.row)
        other.update(id='two',split='validation')
        self.add(other)
        with self.assertRaisesRegex(ValueError,'overlap'):
            eligible_records(self.store,'ref',False)

    def test_invented_group_id_rejected(self):
        self.row['group_id']='different'
        self.add()
        with self.assertRaisesRegex(ValueError,'group identity'):
            eligible_records(self.store,'ref',False)
