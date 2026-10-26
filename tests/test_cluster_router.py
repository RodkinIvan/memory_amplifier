"""cluster/route.py: the hosts file, the jobs file, where a job is placed, the conda env check, and a whole run against a 'host' that is this machine (local=1, a temporary folder as
its root): code and secret copied, jobs started in tmux, results fetched, failures, retries and resuming.

    python -m unittest tests.test_cluster_router -v
"""
import importlib.util
import json
import subprocess
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

from tests import _paths

spec = importlib.util.spec_from_file_location("route", _paths.ROOT / "cluster" / "route.py")
R = importlib.util.module_from_spec(spec)
import sys
sys.modules["route"] = R
spec.loader.exec_module(R)

HOSTS = """
# the hosts of the lab
#router: root=/data/lab per-gpu=3
Host *
    IdentityFile ~/.ssh/id_ed25519_claude
    IdentitiesOnly yes

Host nlp3
    HostName 10.0.0.14
    User ivan
Host nlp2.14
    HostName 10.0.0.43
    User ivan
    #router: per-gpu=2 gpus=0,1 root=/mnt/data/x
Host nlp18
    HostName 10.0.0.18
    #router: skip=1
Host me
    HostName 10.0.0.26
"""


class HostsFile(unittest.TestCase):
    def test_hosts_blocks_and_router_options(self):
        hs = {h.name: h for h in R.parse_hosts(HOSTS, self_addresses={"10.0.0.26"})}
        self.assertEqual(list(hs), ["nlp3", "nlp2.14", "nlp18", "me"])               # `Host *` is ssh's defaults, not a host
        self.assertEqual((hs["nlp3"].hostname, hs["nlp3"].user, hs["nlp3"].root, hs["nlp3"].per_gpu), ("10.0.0.14", "ivan", "/data/lab", 3))   # the options before the first Host apply to all
        self.assertEqual((hs["nlp2.14"].per_gpu, hs["nlp2.14"].gpus, hs["nlp2.14"].root), (2, [0, 1], "/mnt/data/x"))
        self.assertTrue(hs["nlp18"].skip)

    def test_the_host_with_this_machines_address_is_local(self):
        hs = {h.name: h for h in R.parse_hosts(HOSTS, self_addresses={"10.0.0.26"})}
        self.assertTrue(hs["me"].local)
        self.assertFalse(hs["nlp3"].local)

    def test_local_flag(self):
        hs = R.parse_hosts("Host a\n  HostName 1.2.3.4\n  #router: local=1\n", self_addresses=set())
        self.assertTrue(hs[0].local)

    def test_unknown_router_option_and_duplicate_alias_are_errors(self):
        with self.assertRaises(R.RouterError):
            R.parse_hosts("Host a\n  HostName x\n  #router: colour=red\n", self_addresses=set())
        with self.assertRaises(R.RouterError):
            R.parse_hosts("Host a\n  HostName x\nHost a\n  HostName y\n", self_addresses=set())

    def test_the_example_file_parses(self):
        hs = R.parse_hosts((_paths.ROOT / "cluster" / "hosts.example").read_text(), self_addresses=set())
        self.assertTrue(hs and all(h.hostname for h in hs))


class JobsFile(unittest.TestCase):
    def test_spec(self):
        s = R.load_spec(json.dumps({"project": "p", "env": {"file": "cluster/env.yml"}, "sync": ["src"], "secrets": ["tok"], "fetch": ["r/{name}*"],
                                    "jobs": [{"name": "a", "cmd": "x", "gpus": 2, "gb": 7.5, "hosts": ["h"], "retries": 1}]}))
        self.assertEqual((s.project, s.env_file, s.sync, s.secrets, s.jobs[0].gpus, s.jobs[0].gb, s.jobs[0].hosts, s.jobs[0].retries), ("p", "cluster/env.yml", ["src"], ["tok"], 2, 7.5, ["h"], 1))
        self.assertIn("__pycache__", s.exclude)

    def test_errors(self):
        bad = [{}, {"project": "p", "jobs": [{"name": "a"}]}, {"project": "p", "jobs": [{"name": "a", "cmd": "x"}, {"name": "a", "cmd": "y"}]},
               {"project": "p", "jobs": [{"name": "a b", "cmd": "x"}, {"name": "a_b", "cmd": "y"}]}, {"project": "p", "colour": 1}, {"project": "p", "jobs": [{"name": "a", "cmd": "x", "gpu": 1}]}]
        for d in bad:
            with self.assertRaises(R.RouterError, msg=d):
                R.load_spec(json.dumps(d))

    def test_placeholders_and_globs(self):
        self.assertEqual(R.fill("{python} x --gpu {gpu} --n {name} {other} ${HOME}", python="/e/bin/python", gpu="0,1", name="a"), "/e/bin/python x --gpu 0,1 --n a {other} ${HOME}")
        self.assertEqual(R.quote_glob("/a b/./r/x*.json"), "/a' 'b/./r/x*.json")

    def test_env_name_and_hash_change_with_the_post_script(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "e.yml"
            f.write_text("# c\nname: myenv\ndependencies: []\n")
            self.assertEqual(R.env_name(f), "myenv")
            h0 = R.env_hash(f)
            (Path(t) / "e_post.sh").write_text("echo hi\n")
            self.assertNotEqual(R.env_hash(f), h0)
            f.write_text("dependencies: []\n")
            with self.assertRaises(R.RouterError):
                R.env_name(f)

    def test_the_projects_env_file_has_a_name_and_no_local_paths(self):
        f = _paths.ROOT / "cluster" / "env.yml"
        self.assertEqual(R.env_name(f), "memory_amplifier")
        text = f.read_text()
        self.assertNotIn("prefix:", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("flash-attn==", text)                    # installed by env_post.sh (needs torch first)
        self.assertIn("flash-attn", (_paths.ROOT / "cluster" / "env_post.sh").read_text())


class Probe(unittest.TestCase):
    def test_parse(self):
        st = R.parse_probe("GPU 0, 40000, 49140\nGPU 1, 100, 49140\nEXIT a_1 0\nEXIT b 3\nSESSION rt_p_a\nEND\n")
        self.assertEqual(st.gpus[0], (40000 / 1024, 49140 / 1024))
        self.assertEqual(st.exits, {"a_1": 0, "b": 3})
        self.assertEqual(st.sessions, {"rt_p_a"})

    def test_an_incomplete_answer_is_no_answer(self):
        self.assertIsNone(R.parse_probe("GPU 0, 40000, 49140\n"))


def hosts(**kw):
    return {n: R.Host(name=n, per_gpu=kw.get("per_gpu", 2)) for n in kw.get("names", ["a", "b"])}


def state(**gpus):
    """state(a=[(free_gb, total)], ...)"""
    return {n: R.HostState({i: (f, t) for i, (f, t) in enumerate(v)}, {}, set()) for n, v in gpus.items()}


class Placement(unittest.TestCase):
    def go(self, job, hs, st, load=None, reserved=None, headroom=4.0, launched=None, stagger=0.0, host_jobs=None):
        return R.place(job, hs, st, load or {}, reserved or {}, headroom, launched or {}, stagger, 1000.0, host_jobs or {})

    def test_needs_free_memory_plus_headroom(self):
        job = R.Job("j", "x", gpus=1, gb=7)
        self.assertIsNone(self.go(job, hosts(), state(a=[(10.0, 48)], b=[(10.5, 48)])))
        self.assertEqual(self.go(job, hosts(), state(a=[(10.0, 48)], b=[(11.0, 48)])).host, "b")

    def test_prefers_the_least_loaded_gpu_then_the_most_free_memory(self):
        job = R.Job("j", "x", gpus=1, gb=5)
        slot = self.go(job, hosts(), state(a=[(40, 48), (30, 48)], b=[(20, 48)]), load={("a", 0): 1})
        self.assertEqual((slot.host, slot.gpus), ("a", [1]))     # a:1 and b:0 both have no job; a:1 has more free... b:0 has 20, a:1 has 30
        slot = self.go(job, hosts(), state(a=[(40, 48)], b=[(20, 48)]))
        self.assertEqual(slot.host, "a")

    def test_per_gpu_cap(self):
        job = R.Job("j", "x", gpus=1, gb=1)
        self.assertIsNone(self.go(job, hosts(names=["a"], per_gpu=2), state(a=[(40, 48)]), load={("a", 0): 2}))

    def test_reserved_memory_of_jobs_just_started_counts(self):
        job = R.Job("j", "x", gpus=1, gb=10)
        self.assertIsNone(self.go(job, hosts(names=["a"]), state(a=[(30, 48)]), reserved={("a", 0): 17.0}))   # 30 - 17 = 13 < 10 + 4
        self.assertIsNotNone(self.go(job, hosts(names=["a"]), state(a=[(30, 48)]), reserved={("a", 0): 15.0}))

    def test_job_hosts_restriction_and_the_hosts_gpu_list(self):
        job = R.Job("j", "x", gpus=1, gb=1, hosts=["b"])
        self.assertEqual(self.go(job, hosts(), state(a=[(40, 48)], b=[(40, 48)])).host, "b")
        hs = hosts(names=["a"])
        hs["a"].gpus = [1]
        slot = self.go(R.Job("j", "x", gpus=1, gb=1), hs, state(a=[(40, 48), (40, 48)]))
        self.assertEqual(slot.gpus, [1])

    def test_two_gpus_on_one_host(self):
        job = R.Job("j", "x", gpus=2, gb=5)
        self.assertEqual(self.go(job, hosts(), state(a=[(40, 48)], b=[(40, 48), (40, 48)])).host, "b")
        self.assertIsNone(self.go(job, hosts(), state(a=[(40, 48)], b=[(40, 48), (3, 48)])))

    def test_stagger_holds_a_host_back(self):
        job = R.Job("j", "x", gpus=1, gb=1)
        self.assertIsNone(self.go(job, hosts(names=["a"]), state(a=[(40, 48)]), launched={"a": 990.0}, stagger=30.0))
        self.assertIsNotNone(self.go(job, hosts(names=["a"]), state(a=[(40, 48)]), launched={"a": 960.0}, stagger=30.0))

    def test_cpu_jobs_go_to_the_host_with_the_fewest_jobs(self):
        job = R.Job("j", "x", gpus=0)
        self.assertEqual(self.go(job, hosts(), state(a=[], b=[]), host_jobs={"a": 2, "b": 1}).host, "b")

    def test_a_host_that_did_not_answer_is_not_used(self):
        job = R.Job("j", "x", gpus=1, gb=1)
        self.assertIsNone(self.go(job, hosts(names=["a"]), {}))


class FakeConn(R.Conn):
    """A Conn whose commands are answered from a table (substring -> stdout); everything it was asked is kept."""

    def __init__(self, answers, name="h"):
        super().__init__(R.Host(name=name, conda=""), Path("hosts"))
        self.answers, self.asked, self.files = answers, [], []

    def sh(self, script, timeout=120, stdin=None):
        self.asked.append(script)
        if "has-session" in script:               # no session of that name is running
            return subprocess.CompletedProcess(script, 1, "", "")
        for key, out in self.answers.items():
            if key in script:
                return subprocess.CompletedProcess(script, 0, out, "")
        return subprocess.CompletedProcess(script, 0, "", "")

    def rsync(self, args, timeout=7200, cwd=None):
        self.files.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")


class EnvCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Path(self.tmp.name) / "e.yml"
        self.f.write_text("name: myenv\ndependencies: []\n")
        self.sha = R.env_hash(self.f)

    def tearDown(self):
        self.tmp.cleanup()

    def conn(self, env_listed=True, marker="", conda="/opt/conda/bin/conda"):
        answers = {"for c in": conda, "env list --json": json.dumps({"envs": ["/opt/conda", "/opt/conda/envs/myenv"] if env_listed else ["/opt/conda"]}), R.ENV_MARKER: marker}
        return FakeConn(answers)

    def test_existing_env_made_from_this_spec_is_used_silently(self):
        prefix, warnings = R.ensure_env(self.conn(marker=self.sha), self.f, "/p", False, False, 5, log=lambda *a: None)
        self.assertEqual((prefix, warnings), ("/opt/conda/envs/myenv", []))

    def test_existing_env_of_that_name_from_other_files_warns_and_strict_refuses(self):
        _, warnings = R.ensure_env(self.conn(marker="deadbeef"), self.f, "/p", False, False, 5, log=lambda *a: None)
        self.assertEqual(len(warnings), 1)
        self.assertIn("created from other files", warnings[0])
        _, warnings = R.ensure_env(self.conn(marker=""), self.f, "/p", False, False, 5, log=lambda *a: None)
        self.assertIn("not created by this router", warnings[0])
        with self.assertRaises(R.RouterError):
            R.ensure_env(self.conn(marker="deadbeef"), self.f, "/p", False, True, 5, log=lambda *a: None)

    def test_no_conda_is_an_error_that_says_what_to_do(self):
        with self.assertRaisesRegex(R.RouterError, "conda not found"):
            R.ensure_env(self.conn(conda=""), self.f, "/p", False, False, 5, log=lambda *a: None)

    def test_a_missing_env_is_created_from_the_file_and_marked(self):
        c = self.conn(env_listed=False)
        c.answers["env_myenv.exit"] = "0"          # the creation finishes at once
        listed = iter([False, True])

        def env_list(conn, conda, name):             # not there before, there after the creation
            return "/opt/conda/envs/myenv" if next(listed) else ""
        orig, R.env_prefix = R.env_prefix, env_list
        try:
            prefix, warnings = R.ensure_env(c, self.f, "/p", False, False, 5, log=lambda *a: None)
        finally:
            R.env_prefix = orig
        self.assertEqual(prefix, "/opt/conda/envs/myenv")
        started = [s for s in c.asked if "tmux new-session" in s]
        self.assertEqual(len(started), 1)
        self.assertIn("env create -y -n myenv", started[0])
        self.assertIn(self.sha, started[0])                  # the marker is written after the creation
        self.assertIn("env remove -y -n myenv", started[0])  # and a failed creation removes the half-built env
        self.assertTrue(any("e.yml" in f for f in c.files))   # the spec was copied to the host


class WholeRun(unittest.TestCase):
    """A run against this machine as the 'host': its root is a temporary folder, jobs are CPU jobs (gpus = 0) so no GPU is needed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.proj = t / "proj_here"
        (self.proj / "src").mkdir(parents=True)
        (self.proj / "src" / "a.py").write_text("print(1)\n")
        (self.proj / "src" / "__pycache__").mkdir()
        (self.proj / "src" / "__pycache__" / "a.pyc").write_text("x")
        (self.proj / "tok").write_text("secret\n")
        self.hosts_file = t / "hosts"
        self.hosts_file.write_text(f"Host box\n  HostName localhost\n  #router: local=1 root={t / 'hostroot'} per-gpu=2\n")
        self.args = Namespace(headroom=4.0, stagger=0.0, ramp=1.0, poll=0.5, recreate_env=False, strict_env=False, env_timeout=10, dry_run=False)
        self.sessions = []

    def tearDown(self):
        for s in self.sessions:
            subprocess.run(["tmux", "kill-session", "-t", s], capture_output=True)
        self.tmp.cleanup()

    def router(self, jobs, **spec_kw):
        d = {"project": "demo", "sync": ["src"], "secrets": ["tok"], "fetch": ["out/{name}.txt"], "jobs": jobs, **spec_kw}
        hosts = [h for h in R.parse_hosts(self.hosts_file.read_text(), self_addresses=set())]
        r = R.Router(R.load_spec(json.dumps(d)), hosts, self.hosts_file, self.proj, self.proj.parent / "jobs.state.json", self.args, log=lambda *a: None)
        self.sessions += [r.session(j) for j in r.spec.jobs]
        return r

    def job(self, name, cmd=None, **kw):
        return {"name": name, "cmd": cmd or f"mkdir -p out && echo hi-{name} > out/{name}.txt", "gpus": 0, "expect": [f"out/{name}.txt"], **kw}

    def test_code_secret_and_results(self):
        r = self.router([self.job("a"), self.job("b")])
        r.prepare()
        rd = Path(r.ctx["box"].project_dir)
        self.assertTrue((rd / "src" / "a.py").exists())
        self.assertFalse((rd / "src" / "__pycache__").exists())                 # excluded
        self.assertEqual(oct((rd / "tok").stat().st_mode & 0o777), "0o600")
        self.assertEqual(r.run(), 0)
        self.assertEqual((self.proj / "out" / "a.txt").read_text().strip(), "hi-a")
        self.assertEqual((self.proj / "out" / "b.txt").read_text().strip(), "hi-b")
        self.assertTrue((self.proj / "logs" / "router" / "a.log").exists())     # the job's log comes back too
        st = json.loads((self.proj.parent / "jobs.state.json").read_text())["jobs"]
        self.assertEqual({k: v["status"] for k, v in st.items()}, {"a": "done", "b": "done"})

    def test_a_failing_job_is_reported_and_does_not_stop_the_others(self):
        r = self.router([self.job("bad", "exit 3"), self.job("ok")])
        r.prepare()
        self.assertEqual(r.run(), 1)
        self.assertEqual((r.state["jobs"]["bad"]["status"], r.state["jobs"]["bad"]["exit"]), ("failed", 3))
        self.assertEqual(r.state["jobs"]["ok"]["status"], "done")

    def test_a_missing_expected_file_makes_the_job_fail(self):
        r = self.router([self.job("noout", "true")])
        r.prepare()
        self.assertEqual(r.run(), 1)
        self.assertIn("expected file missing", " ".join(r.state["jobs"]["noout"]["problems"]))

    def test_retries(self):
        cmd = "test -f once || { touch once; exit 1; }; mkdir -p out && echo ok > out/flaky.txt"
        r = self.router([self.job("flaky", cmd, retries=1)])
        r.prepare()
        self.assertEqual(r.run(), 0)
        self.assertEqual(r.state["jobs"]["flaky"]["attempt"], 2)

    def test_finished_jobs_are_not_run_again(self):
        r = self.router([self.job("a"), self.job("b")])
        r.state["jobs"]["a"] = {"status": "done", "host": "box", "gpus": [], "attempt": 1, "exit": 0}
        r.prepare()
        self.assertEqual(r.run(), 0)
        self.assertFalse((self.proj / "out" / "a.txt").exists())                # a was recorded as done: not started
        self.assertTrue((self.proj / "out" / "b.txt").exists())

    def test_a_running_job_is_adopted_after_a_restart(self):
        r1 = self.router([self.job("slow", "sleep 4; mkdir -p out && echo late > out/slow.txt")])
        r1.prepare()
        r1.start(r1.spec.jobs[0], R.Slot("box", []))                           # the first router starts it and goes away
        r2 = self.router([self.job("slow", "sleep 4; mkdir -p out && echo late > out/slow.txt")])
        r2.state = json.loads((self.proj.parent / "jobs.state.json").read_text())
        r2.prepare()
        self.assertEqual(r2.run(), 0)                                           # the second finds the session, waits, fetches
        self.assertEqual((self.proj / "out" / "slow.txt").read_text().strip(), "late")
        self.assertEqual(r2.state["jobs"]["slow"]["attempt"], 1)                # started once

    def test_a_job_the_host_has_no_room_for_stops_the_run_with_a_message(self):
        r = self.router([{"name": "big", "cmd": "true", "gpus": 64}])
        r.prepare()
        with self.assertRaisesRegex(R.RouterError, "no host can run"):
            r.run()

    def test_dry_run_starts_nothing(self):
        self.args.dry_run = True
        r = self.router([self.job("a")])
        r.prepare()
        self.assertEqual(r.run(dry=True), 0)
        self.assertNotIn("a", r.state["jobs"])


if __name__ == "__main__":
    unittest.main()
