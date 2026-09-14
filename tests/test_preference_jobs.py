import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from backend.preference.jobs import gpu_lock, launch_exclusive, Timeout


class JobLockTests(unittest.TestCase):
    def test_other_launch_is_refused_and_failed_launch_releases_lock(self):
        with gpu_lock().acquire(timeout=0):
            with self.assertRaises(Timeout):
                launch_exclusive([sys.executable,'-c','pass'])
        with patch('backend.preference.jobs.subprocess.Popen',side_effect=OSError('test launch failure')):
            with self.assertRaises(OSError):
                launch_exclusive(['not-an-executable'])
        with gpu_lock().acquire(timeout=0):
            pass

    def test_child_completion_releases_lock(self):
        process=launch_exclusive([sys.executable,'-c','import time; time.sleep(.2)'])
        with self.assertRaises(Timeout):
            gpu_lock().acquire(timeout=0)
        self.assertEqual(process.wait(timeout=5),0)
        with gpu_lock().acquire(timeout=2):
            pass
