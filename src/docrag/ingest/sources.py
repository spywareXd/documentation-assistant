"""Fetch documentation sources from GitHub at pinned commits.

Each repo is shallow sparse-cloned once into data/corpus/repos/. The commit is
recorded in sources.lock.json so later runs (and other machines) build the
exact same corpus.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from docrag.config import ROOT, Config
from docrag.logger import log_info, log_success, log_warning

LOCK_FILE = ROOT / "sources.lock.json"


class SourceSpec(BaseModel):
    name: str
    repo: str
    branch: str = "main"
    license: str = "unknown"
    flavor: str  # mkdocs | mintlify | hugo
    docs_path: str
    extra_paths: list[str] = []
    include: list[str] = []
    exclude: list[str] = []
    url: str


@dataclass
class RawPage:
    source: str
    rel_path: str  # path relative to docs_path, posix
    abs_path: Path
    repo_dir: Path
    url: str
    commit: str
    flavor: str


def load_sources(cfg: Config) -> list[SourceSpec]:
    data = yaml.safe_load(cfg.path(cfg.corpus.sources_file).read_text(encoding="utf-8"))
    specs = [SourceSpec(**s) for s in data["sources"]]
    if cfg.corpus.sources:
        specs = [s for s in specs if s.name in cfg.corpus.sources]
    return specs


# --------------------------------------------------------------------------- glob
@lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> re.Pattern:
    """Glob -> regex where `*` stays inside one path segment and `**/` spans any depth."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, pattern: str) -> bool:
    return bool(_glob_regex(pattern).match(path))


def page_url(template: str, rel_path: str) -> str:
    stem = re.sub(r"\.(md|mdx)$", "", rel_path)
    parts = stem.split("/")
    if parts[-1] in ("index", "_index"):
        parts = parts[:-1]
    path = "/".join(parts)
    url = template.replace("{path}", path)
    return re.sub(r"(?<!:)//+", "/", url)  # collapse "//" left by an empty path


# --------------------------------------------------------------------------- git
def _git(args: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


def _read_lock() -> dict[str, str]:
    return json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}


def _write_lock(lock: dict[str, str]) -> None:
    LOCK_FILE.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n")


def ensure_repo(repo: str, branch: str, paths: list[str], repos_dir: Path, lock: dict[str, str]) -> tuple[Path, str]:
    """Sparse, shallow checkout of `paths` at the locked commit (or branch head on first run)."""
    target = repos_dir / repo.replace("/", "__")
    url = f"https://github.com/{repo}.git"
    if not target.exists():
        log_info(f"Cloning {repo} ({branch}) sparse: {paths}")
        repos_dir.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--filter=blob:none", "--no-checkout", "--depth", "1", "--branch", branch, url, str(target)])
    _git(["sparse-checkout", "set", "--no-cone", *[f"/{p.strip('/')}/" for p in paths]], cwd=target)

    pinned = lock.get(repo)
    if pinned:
        head = _git(["rev-parse", "HEAD"], cwd=target) if (target / ".git" / "HEAD").exists() else ""
        if head != pinned:
            _git(["fetch", "--depth", "1", "origin", pinned], cwd=target)
        _git(["checkout", "--quiet", pinned], cwd=target)
        commit = pinned
    else:
        _git(["checkout", "--quiet", branch], cwd=target)
        commit = _git(["rev-parse", "HEAD"], cwd=target)
        lock[repo] = commit
        log_success(f"Pinned {repo} at {commit[:10]}")
    return target, commit


def fetch_pages(cfg: Config) -> list[RawPage]:
    specs = load_sources(cfg)
    repos_dir = cfg.path(cfg.corpus.data_dir) / "repos"
    lock = _read_lock()

    # one checkout per repo, covering every path any source needs from it
    needed: dict[str, tuple[str, set[str]]] = {}
    for s in specs:
        branch, paths = needed.setdefault(s.repo, (s.branch, set()))
        paths.update([s.docs_path, *s.extra_paths])
    checkouts = {repo: ensure_repo(repo, br, sorted(p), repos_dir, lock) for repo, (br, p) in needed.items()}
    _write_lock(lock)

    pages: list[RawPage] = []
    for s in specs:
        repo_dir, commit = checkouts[s.repo]
        docs_root = repo_dir / s.docs_path
        if not docs_root.exists():
            log_warning(f"{s.name}: {s.docs_path} not found in {s.repo}")
            continue
        count = 0
        for f in sorted(docs_root.rglob("*")):
            if f.suffix not in (".md", ".mdx") or not f.is_file():
                continue
            rel = f.relative_to(docs_root).as_posix()
            if s.include and not any(glob_match(rel, p) for p in s.include):
                continue
            if any(glob_match(rel, p) for p in s.exclude):
                continue
            pages.append(RawPage(s.name, rel, f, repo_dir, page_url(s.url, rel), commit, s.flavor))
            count += 1
        log_info(f"{s.name}: {count} pages")
    return pages
