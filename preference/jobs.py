"""One GPU job at a time through this checkout's supported launchers."""
import subprocess
from pathlib import Path
from threading import Thread

from filelock import FileLock, Timeout


def gpu_lock():
    path = Path(__file__).resolve().parents[1] / 'runtime_store' / 'preference_gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path), thread_local=False)


def launch_exclusive(command):
    lock = gpu_lock()
    lock.acquire(timeout=0)
    try:
        process = subprocess.Popen(command)
    except BaseException:
        lock.release()
        raise

    def release_when_finished():
        try:
            process.wait()
        finally:
            lock.release()

    Thread(target=release_when_finished, daemon=True).start()
    return process
