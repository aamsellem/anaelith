"""--crash-resume: an engine that dies mid-request is restarted and the answer goes on after its written tokens."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from serve.frontend import ChatTemplate
from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine
from serve.service import Service
from serve.util.text import ByteTokenizer

# Every start appends one line to `starts`; every GEN line is kept in `gens`. A start dies at its first request after
# writing "Hi" while it is one of the first `--crashes` starts; later starts answer "!" and stop.
FAKE = r'''
import sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--root') + 1])
crashes = int(sys.argv[sys.argv.index('--crashes') + 1])
starts = root / 'starts'
with open(starts, 'a') as f:
    f.write('start\n')
start_n = len(starts.read_text().splitlines())
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith(('GEN', 'BGEN')):
        with open(root / 'gens', 'a') as f:
            f.write(line)
        if start_n <= crashes:
            print('T 72\nT 105', flush=True)
            print('ERR verify: timed out at layer 47; device plan: ring 47 published, host flag 47', flush=True)
            print('DONE 0 1 0 1 error', flush=True)
            continue
        print('T 33\nDONE 1 1 0 1 stop', flush=True)
'''


class CrashResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        py = self.root / 'native.py'
        if os.name == 'nt':
            py.write_text(FAKE)
            self.script = self.root / 'native.cmd'
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script = py
            py.write_text('#!' + sys.executable + '\n' + FAKE)
            py.chmod(0o700)

    def service(self, crashes, resume):
        e = ProcessEngine(str(self.script), ['--root', str(self.root), '--crashes', str(crashes)])
        svc = Service(e, ByteTokenizer(),
                      ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        svc.crash_resume = resume
        self.addCleanup(lambda: svc.engine.close())
        return svc

    def run_request(self, svc):
        out = list(svc.run([1, 2], False, [], 8, {}, threading.Event()))
        text = "".join(ev.text or "" for kind, ev in out if kind == "event" and ev.kind == "content")
        return text, out[-1][1]

    def gens(self):
        return [line.split()[-1] for line in (self.root / 'gens').read_text().splitlines()]

    def test_the_answer_goes_on_after_the_written_tokens(self):
        svc = self.service(crashes=1, resume=2)
        text, done = self.run_request(svc)
        self.assertEqual(text, "Hi!")
        self.assertEqual(done["completion_tokens"], 3)
        self.assertNotEqual(svc.history[-1]["finish"], "error")
        # the second start reads the prompt and the two tokens written before the death
        self.assertEqual(self.gens(), ["1,2", "1,2,72,105"])
        self.assertEqual(len((self.root / 'starts').read_text().splitlines()), 2)

    def test_resumes_are_bounded(self):
        svc = self.service(crashes=5, resume=1)
        with self.assertRaises(EngineDied):
            self.run_request(svc)
        self.assertEqual(svc.history[-1]["finish"], "error")
        self.assertEqual(len(self.gens()), 2)            # the first try and one resume, no third

    def test_off_by_default(self):
        svc = self.service(crashes=1, resume=0)
        with self.assertRaises(EngineDied):
            self.run_request(svc)
        self.assertEqual(len(self.gens()), 1)


if __name__ == "__main__":
    unittest.main()


class CrashResumeParallel(unittest.TestCase):
    """The same in parallel mode (--parallel N), where every request runs in a batch slot."""

    def setUp(self):
        import subprocess
        from unittest import mock
        from serve.handler import serve
        from serve.tests.test_parallel import FAKE_BATCH
        bt = '        print(f"BT {b} " + " ".join(map(str, got)), flush=True)\n'
        assert FAKE_BATCH.count(bt) == 1
        # the engine's first start dies inside the first reply, once a few of its tokens are out
        tt = '            out += 1\n'
        assert FAKE_BATCH.count(tt) == 1
        death = '''if die and not os.path.exists(die) and {n} >= 4:
{i}    open(die, "w").write("died")
{i}    print("ERR verify: timed out at layer 47; device plan: ring 47 published, host flag 47", flush=True)
{i}    time.sleep(30)
'''
        # the engine's first start dies inside the first reply, once a few of its tokens are out (slot or solo run)
        fake = FAKE_BATCH.replace(bt, bt + "        " + death.format(n="produced", i="        "))
        fake = fake.replace(tt, tt + "            " + death.format(n="out", i="            "))
        fake = fake.replace("import queue, sys, threading, time\n", "import os, queue, sys, threading, time\n", 1)
        fake = fake.replace('args = sys.argv[1:]\n', 'args = sys.argv[1:]\ndie = args[args.index("--die-once") + 1] '
                                                     'if "--die-once" in args else None\n', 1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        script = root / "fake_engine.py"
        script.write_text(fake, encoding="utf-8")
        self.log = root / "requests.log"
        real = subprocess.Popen
        # kept for the whole test: the restart spawns the engine again
        patch = mock.patch.object(subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine = ProcessEngine("basalt", ["--batch", "2", "--log", str(self.log), "--die-once", str(root / "died")])
        self.svc = Service(self.engine, ByteTokenizer(),
                           ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        self.svc.crash_resume = 1
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(lambda: (self.httpd.shutdown(), self.httpd.server_close(), self.svc.engine.unload()))
        self.died = root / "died"

    def test_a_slot_resumes_after_the_engine_dies(self):
        import json
        import urllib.request
        body = {"model": "basalt", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
                "reasoning_effort": "none"}
        req = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read().decode())
        self.assertTrue(self.died.exists(), "the fake engine was meant to die once")
        self.assertEqual(out["choices"][0]["message"]["content"], "ok, done.")
        self.assertNotEqual(self.svc.history[-1]["finish"], "error")
        runs = [x.split()[0] for x in self.log.read_text().splitlines() if x.startswith(("GEN", "BGEN", "SGEN"))]
        self.assertEqual(len(runs), 2, runs)            # the run that died and the resumed one
