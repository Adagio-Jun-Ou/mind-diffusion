"""Terminal world simulator - synthetic shell sessions with learnable dynamics.

Research-backed design (CLI-Gym-style environment + error/recovery pairs):
a small virtual filesystem, a command grammar with slots sampled from state,
deterministic outputs, realistic error taxonomy (ENOENT, EEXIST, EISDIR,
command-not-found) and recovery patterns. Sessions are the training data for
MindJEPA: world dynamics that a latent model can actually learn.

    sim = VirtualShell(seed=0)
    turns = generate_session(sim, rng, n_turns=12)
    # turns: [{"cmd": "ls", "output": "src tests", "exit": 0}, ...]

    obs, acts = TerminalWorldAdapter().session_to_frames(turns_as_pairs)

Also provides:
  normalize_volatile(text)  - strip timestamps/pids/paths for real logs
  load_jsonl_sessions(path) - ingest real (cmd, output) logs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("terminal_world")

MAX_LINES = 12


def normalize_volatile(text: str) -> str:
    """Strip tokens that vary run-to-run but carry no world dynamics."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TS>", text)
    text = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<TS>", text)
    text = re.sub(r"\bpid[ =]\d+\b", "pid <N>", text, flags=re.I)
    text = re.sub(r"\b\d+\.\d+\s*(ms|s|sec)\b", "<DUR>", text)
    text = re.sub(r"[A-Za-z]:\\[^\s]+|\b/(?:home|tmp|usr)/[^\s]+", "<PATH>", text)
    return text


class VirtualShell:
    """Minimal deterministic shell over an in-memory filesystem."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.root = "/home/user/project"
        self._reset()

    def _reset(self):
        self.dirs = {self.root, self.root + "/src", self.root + "/tests",
                     self.root + "/data", self.root + "/src/utils"}
        self.files = {
            f"{self.root}/README.md": "# project\nA small project.",
            f"{self.root}/setup.py": "from setuptools import setup\nsetup(name='proj')",
            f"{self.root}/src/main.py": "def main():\n    print('hello')",
            f"{self.root}/src/utils/io.py": "def load(p):\n    return open(p).read()",
            f"{self.root}/tests/test_main.py": "def test_main():\n    assert True",
            f"{self.root}/data/notes.txt": "alpha\nbeta\ngamma",
        }
        self.cwd = self.root
        self.git_branch = "main"
        self.git_commit = 3
        self.staged: set = set()
        self.dirty: set = set()

    # -- path helpers -------------------------------------------------------- #
    def _abs(self, p: str) -> str:
        if p.startswith("/"):
            parts = [x for x in p.split("/") if x]
        else:
            parts = [x for x in (self.cwd.split("/") + p.split("/")) if x and x != "."]
        out = []
        for x in parts:
            if x == "..":
                if out:
                    out.pop()
            else:
                out.append(x)
        return "/" + "/".join(out)

    def _is_dir(self, p: str) -> bool:
        return p in self.dirs

    def _entries(self, p: str):
        pref = p.rstrip("/") + "/"
        names = set()
        for d in self.dirs:
            if d.startswith(pref) and d != p:
                names.add(d[len(pref):].split("/")[0] + "/")
        for f in self.files:
            if f.startswith(pref):
                names.add(f[len(pref):].split("/")[0])
        return sorted(names)

    # -- command execution ---------------------------------------------------- #
    def run(self, cmd: str):
        cmd = cmd.strip()
        if not cmd:
            return "", 0
        parts = cmd.split()
        head = parts[0]
        try:
            if head == "pwd":
                return self.cwd, 0
            if head == "ls":
                p = self._abs(parts[1]) if len(parts) > 1 else self.cwd
                if not self._is_dir(p):
                    return f"ls: cannot access '{parts[1] if len(parts) > 1 else p}': No such file or directory", 2
                return "  ".join(self._entries(p)), 0
            if head == "cd":
                if len(parts) < 2:
                    self.cwd = self.root
                    return "", 0
                p = self._abs(parts[1])
                if not self._is_dir(p):
                    return f"cd: no such file or directory: {parts[1]}", 1
                self.cwd = p
                return "", 0
            if head == "cat":
                if len(parts) < 2:
                    return "cat: missing operand", 1
                p = self._abs(parts[1])
                if self._is_dir(p):
                    return f"cat: {parts[1]}: Is a directory", 1
                if p not in self.files:
                    return f"cat: {parts[1]}: No such file or directory", 1
                return self.files[p], 0
            if head == "mkdir":
                if len(parts) < 2:
                    return "mkdir: missing operand", 1
                p = self._abs(parts[-1])
                if p in self.dirs:
                    return f"mkdir: cannot create directory '{parts[-1]}': File exists", 1
                self.dirs.add(p)
                return "", 0
            if head == "touch":
                if len(parts) < 2:
                    return "touch: missing file operand", 1
                p = self._abs(parts[-1])
                if p not in self.dirs and p not in self.files:
                    self.files[p] = ""
                return "", 0
            if head == "rm":
                if len(parts) < 2:
                    return "rm: missing operand", 1
                rec = any(a in ("-r", "-rf", "-fr") for a in parts)
                p = self._abs(parts[-1])
                if p in self.dirs:
                    if not rec:
                        return f"rm: cannot remove '{parts[-1]}': Is a directory", 1
                    pref = p.rstrip("/") + "/"
                    self.dirs = {d for d in self.dirs if d != p and not d.startswith(pref)}
                    self.files = {f: c for f, c in self.files.items()
                                  if not f.startswith(pref)}
                    self.dirty.add(p)
                    return "", 0
                if p in self.files:
                    del self.files[p]
                    self.dirty.add(p)
                    return "", 0
                return f"rm: cannot remove '{parts[-1]}': No such file or directory", 1
            if head == "echo" and ">" in parts:
                i = parts.index(">")
                text = " ".join(parts[1:i])
                p = self._abs(parts[i + 1])
                self.files[p] = text
                self.dirty.add(p)
                return "", 0
            if head == "grep":
                if len(parts) < 3:
                    return "usage: grep PATTERN FILE", 2
                p = self._abs(parts[-1])
                if p not in self.files:
                    return f"grep: {parts[-1]}: No such file or directory", 2
                hits = [ln for ln in self.files[p].splitlines() if parts[1] in ln]
                return "\n".join(hits), 0 if hits else 1
            if head == "python":
                if len(parts) < 2:
                    return "usage: python FILE", 2
                p = self._abs(parts[1])
                if p not in self.files:
                    return f"python: can't open file '{parts[1]}': [Errno 2] No such file or directory", 2
                return "hello", 0
            if head == "pytest":
                n = sum(1 for f in self.files if f.endswith(".py") and "test" in f)
                if n == 0:
                    return "no tests ran", 5
                return f"{n} passed in 0.12s", 0
            if head == "git":
                return self._git(parts[1:])
            return f"{head}: command not found", 127
        except Exception as exc:  # simulator bugs should look like shell errors
            return f"{head}: error: {exc}", 1

    def _git(self, args):
        sub = args[0] if args else "status"
        if sub == "status":
            lines = [f"On branch {self.git_branch}"]
            if self.staged:
                lines.append("Changes to be committed:")
                lines += [f"  modified: {s}" for s in sorted(self.staged)]
            elif self.dirty:
                lines += [f"  modified: {s}" for s in sorted(self.dirty)]
            else:
                lines.append("nothing to commit, working tree clean")
            return "\n".join(lines), 0
        if sub == "add":
            if not self.dirty:
                return "Nothing specified, nothing added.", 0
            self.staged |= self.dirty
            self.dirty = set()
            return "", 0
        if sub == "commit":
            if not self.staged:
                return "nothing to commit, working tree clean", 1
            self.git_commit += 1
            msg = " ".join(a for a in args[2:] if not a.startswith("-")).strip('"') or "update"
            n = len(self.staged)
            self.staged = set()
            return f"[{self.git_branch} {self.git_commit:04x}] {msg}\n {n} files changed", 0
        if sub == "log":
            return "\n".join(f"commit {i:04x} update" for i in range(self.git_commit, 0, -1)), 0
        return f"git: '{sub}' is not a git command", 1


# --------------------------------------------------------------------------- #
# Session generation policy: plausible moves + error recovery
# --------------------------------------------------------------------------- #
def _choose_command(sim: VirtualShell, rng: np.random.Generator, last_failed: bool) -> str:
    files = sorted(sim.files)
    dirs = sorted(d for d in sim.dirs if d != sim.cwd)
    pick_file = files[int(rng.integers(0, len(files)))] if files else "notes.txt"
    pick_dir = dirs[int(rng.integers(0, len(dirs)))] if dirs else "src"

    if last_failed and rng.random() < 0.6:
        # recovery policy: look around, then retry something safe
        return ["ls", f"ls {sim.cwd}", "pwd", "git status"][int(rng.integers(0, 4))]

    # deliberate faults: error/recovery dynamics are the key learnable signal
    if rng.random() < 0.18:
        absent = ["missing.txt", "src/nope.py", "data/old.csv",
                  "nope", f"{pick_dir}/ghost"][int(rng.integers(0, 5))]
        return ["cat " + absent, "cd " + absent, "rm " + absent,
                "python README.md", "grep xyz " + absent,
                "mkdir src"][int(rng.integers(0, 6))]

    roll = rng.random()
    if roll < 0.18:
        return "ls"
    if roll < 0.28:
        return f"cd {pick_dir}"
    if roll < 0.40:
        return f"cat {pick_file}"
    if roll < 0.48:
        return f"mkdir {pick_dir}/build" if not rng.integers(0, 2) else "mkdir build"
    if roll < 0.55:
        return f"touch src/tmp_{int(rng.integers(0, 99))}.py"
    if roll < 0.62:
        return f"grep {['alpha', 'def', 'setup', 'import'][int(rng.integers(0, 4))]} {pick_file}"
    if roll < 0.70:
        return f"echo {' '.join(str(rng.integers(0, 99)) for _ in range(3))} > data/out.txt"
    if roll < 0.78:
        return f"python {pick_file if pick_file.endswith('.py') else 'src/main.py'}"
    if roll < 0.86:
        return "pytest" if rng.integers(0, 2) else "python tests/test_main.py"
    if roll < 0.94:
        return ["git status", "git add .", "git commit -m update"][int(rng.integers(0, 3))]
    return f"rm {pick_file}" if rng.integers(0, 2) else f"rm -r {pick_dir}/build"


def generate_session(sim: VirtualShell, rng: np.random.Generator, n_turns: int = 12):
    turns = []
    last_failed = False
    for _ in range(n_turns):
        cmd = _choose_command(sim, rng, last_failed)
        output, code = sim.run(cmd)
        lines = output.splitlines() or [""]
        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES] + ["..."]
        turns.append({"cmd": cmd, "output": "\n".join(lines), "exit": int(code)})
        last_failed = code != 0
    return turns


def generate_sessions(n: int = 64, seed: int = 0, min_turns: int = 8,
                      max_turns: int = 20):
    """Many independent sessions, each a fresh virtual machine."""
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        sim = VirtualShell(seed=seed + i)
        length = int(rng.integers(min_turns, max_turns + 1))
        out.append(generate_session(sim, rng, n_turns=length))
    return out


def turns_to_pairs(turns):
    return [(t["cmd"], t["output"]) for t in turns]


def load_jsonl_sessions(path: str, max_sessions: int = 0):
    """Ingest real sessions: JSONL rows of {cmd, output, exit?} grouped by session_id."""
    grouped = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("session_id", 0)
            grouped.setdefault(sid, []).append({
                "cmd": normalize_volatile(str(row.get("cmd", ""))),
                "output": normalize_volatile(str(row.get("output", ""))),
                "exit": int(row.get("exit", 0)),
            })
    sessions = [grouped[k] for k in sorted(grouped.keys(), key=str)]
    if max_sessions:
        sessions = sessions[:max_sessions]
    log.info("loaded %d sessions (%d turns) from %s",
             len(sessions), sum(len(s) for s in sessions), path)
    return sessions


if __name__ == "__main__":
    sessions = generate_sessions(n=3, seed=1)
    turns = turns_to_pairs(sessions[0])
    print(f"session 0: {len(turns)} turns")
    for cmd, out in turns[:6]:
        print(f"$ {cmd}\n{out}\n")
