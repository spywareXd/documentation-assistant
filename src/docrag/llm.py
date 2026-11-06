"""Groq (GPT-OSS) access with caching, rate limiting, retries and token accounting.

Every call goes through `LLM.chat()`:
- responses are cached in SQLite keyed by (model, messages, params), so re-running an
  experiment costs nothing and is deterministic;
- a client-side limiter keeps us under Groq's requests-per-minute limit;
- 429 / 5xx are retried with exponential backoff;
- token usage is counted per role so eval runs can report cost per 1k queries.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from docrag.config import Config

load_dotenv()

Role = Literal["generator", "utility", "judge"]
T = TypeVar("T", bound=BaseModel)


class _Limiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / max(per_minute, 1)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            time.sleep(delay)


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, response TEXT, usage TEXT)")
        self.lock = threading.Lock()

    def get(self, key: str) -> tuple[str, dict] | None:
        with self.lock:
            row = self.db.execute("SELECT response, usage FROM calls WHERE key=?", (key,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def put(self, key: str, response: str, usage: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?)", (key, response, json.dumps(usage)))
            self.db.commit()


def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status in (408, 409, 429, 500, 502, 503, 504) or "rate limit" in str(exc).lower()


class LLM:
    def __init__(self, cfg: Config):
        from groq import Groq  # installed with langchain-groq

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        self.cfg = cfg.llm
        self.client = Groq(max_retries=0)  # retries are handled here, with backoff
        self.cache = _Cache(cfg.path(cfg.llm.cache_dir) / "groq.sqlite")
        self.limiter = _Limiter(cfg.llm.requests_per_minute)
        self.usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._usage_lock = threading.Lock()

    def model_for(self, role: Role) -> str:
        return getattr(self.cfg, role)

    # ------------------------------------------------------------------ core call
    def chat(
        self,
        messages: list[dict],
        role: Role = "generator",
        json_mode: bool = False,
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> str:
        model = self.model_for(role)
        params = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_completion_tokens": max_tokens,
        }
        if "gpt-oss" in model:
            params["reasoning_effort"] = self.cfg.reasoning_effort
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if use_cache and (hit := self.cache.get(key)):
            text, usage = hit
            self._count(role, usage, cached=True)
            return text

        text, usage = self._call(params)
        self.cache.put(key, text, usage)
        self._count(role, usage, cached=False)
        return text

    def _call(self, params: dict) -> tuple[str, dict]:
        @retry(
            retry=retry_if_exception(_retryable),
            wait=wait_exponential_jitter(initial=2, max=60),
            stop=stop_after_attempt(self.cfg.max_retries),
            reraise=True,
        )
        def go():
            self.limiter.wait()
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(**params)
            u = resp.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            return resp.choices[0].message.content or "", usage

        return go()

    def _count(self, role: str, usage: dict, cached: bool) -> None:
        with self._usage_lock:
            u = self.usage[role]
            u["calls"] += 1
            u["cached_calls"] += int(cached)
            u["prompt_tokens"] += usage.get("prompt_tokens", 0)
            u["completion_tokens"] += usage.get("completion_tokens", 0)

    # ------------------------------------------------------------------ helpers
    def structured(self, messages: list[dict], schema: type[T], role: Role = "utility", max_tokens: int = 1024) -> T:
        """JSON-mode call parsed into a pydantic model; one repair attempt on bad JSON."""
        sys_hint = {
            "role": "system",
            "content": "Respond with a single JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema()),
        }
        msgs = [sys_hint, *messages]
        text = self.chat(msgs, role=role, json_mode=True, max_tokens=max_tokens)
        try:
            return schema.model_validate_json(text)
        except ValidationError as err:
            repair = [*msgs, {"role": "assistant", "content": text},
                      {"role": "user", "content": f"That JSON was invalid: {err.errors()[:3]}. Return corrected JSON only."}]
            return schema.model_validate_json(self.chat(repair, role=role, json_mode=True, max_tokens=max_tokens))

    def usage_report(self) -> dict:
        return {role: dict(v) for role, v in self.usage.items()}


_instance: LLM | None = None


def get_llm(cfg: Config) -> LLM:
    global _instance
    if _instance is None or _instance.cfg != cfg.llm:
        _instance = LLM(cfg)
    return _instance
