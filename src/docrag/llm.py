"""Groq access with caching, rate limiting, retries and token accounting.

Every call goes through `LLM.chat()`:
- responses are cached in SQLite keyed by (model, messages, params), so re-running an
  experiment costs nothing and is deterministic;
- a per-model limiter paces requests AND tokens per minute (Groq free tier: 30 RPM, 8k TPM);
- short-term 429 / 5xx are retried with exponential backoff;
- a spent daily quota (tokens/requests per day) raises QuotaExhausted immediately, so a run
  stops cleanly and the next run resumes from the cache;
- token usage is counted per role so eval runs can report tokens and cost per query.
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


class QuotaExhausted(RuntimeError):
    """The model's daily token/request quota is used up. Re-run later; finished calls are cached."""


class _Limiter:
    """Sliding 60 s window over requests and tokens for one model."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm, self.tpm = rpm, tpm
        self.events: list[tuple[float, int]] = []  # (time, tokens)
        self.lock = threading.Lock()

    def _trim(self, now: float) -> None:
        self.events = [(t, n) for t, n in self.events if now - t < 60]

    def wait(self, est_tokens: int) -> None:
        est_tokens = min(est_tokens, int(self.tpm * 0.9))
        while True:
            with self.lock:
                now = time.monotonic()
                self._trim(now)
                used = sum(n for _, n in self.events)
                if len(self.events) < self.rpm and used + est_tokens <= self.tpm * 0.9:
                    self.events.append((now, est_tokens))
                    return
                wait = 60 - (now - self.events[0][0]) + 0.1 if self.events else 1.0
            time.sleep(max(wait, 0.5))

    def correct(self, est_tokens: int, actual: int) -> None:
        """Replace the estimate with the real usage once the response arrives."""
        with self.lock:
            for i in range(len(self.events) - 1, -1, -1):
                if self.events[i][1] == est_tokens:
                    self.events[i] = (self.events[i][0], actual)
                    return


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


def _is_daily_quota(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "per day" in msg and ("tpd" in msg or "rpd" in msg or "rate limit" in msg)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, QuotaExhausted):
        return False
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
        self.limiters: dict[str, _Limiter] = {}  # each Groq model has its own quota
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
            model = params["model"]
            limiter = self.limiters.setdefault(model, _Limiter(self.cfg.requests_per_minute, self.cfg.tokens_per_minute))
            est = len(json.dumps(params["messages"])) // 3 + min(params["max_completion_tokens"], 800)
            limiter.wait(est)
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**params)
            except Exception as err:
                if _is_daily_quota(err):
                    raise QuotaExhausted(f"{model}: daily quota used up. {str(err)[:300]}") from err
                raise
            u = resp.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            limiter.correct(est, usage["prompt_tokens"] + usage["completion_tokens"])
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
        try:
            text = self.chat(msgs, role=role, json_mode=True, max_tokens=max_tokens)
        except Exception as err:
            if isinstance(err, QuotaExhausted) or "json_validate_failed" not in str(err):
                raise
            # Groq rejects non-JSON output in JSON mode; retry in plain mode and extract the object
            strict = [*msgs, {"role": "user", "content": "Reply with the JSON object only: no prose, no code fences."}]
            text = _extract_json(self.chat(strict, role=role, json_mode=False, max_tokens=max_tokens))
        try:
            return schema.model_validate_json(text)
        except ValidationError as err:
            repair = [*msgs, {"role": "assistant", "content": text},
                      {"role": "user", "content": f"That JSON was invalid: {err.errors()[:3]}. Return corrected JSON only."}]
            return schema.model_validate_json(
                _extract_json(self.chat(repair, role=role, json_mode=False, max_tokens=max_tokens)))

    def usage_report(self) -> dict:
        return {role: dict(v) for role, v in self.usage.items()}


def _extract_json(text: str) -> str:
    """Outermost {...} in a reply that may carry prose or code fences around it."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


_instance: LLM | None = None


def get_llm(cfg: Config) -> LLM:
    global _instance
    if _instance is None or _instance.cfg != cfg.llm:
        _instance = LLM(cfg)
    return _instance
