from docrag.llm import QuotaExhausted, _extract_json, _is_daily_quota, _Limiter, _retryable

DAILY = ("Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per day (TPD): "
         "Limit 200000, Used 199858, Requested 306. Please try again in 1m10s.")
PER_MINUTE = "Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per minute (TPM): Limit 8000"


def test_daily_quota_is_not_retried_but_per_minute_is():
    assert _is_daily_quota(Exception(DAILY))
    assert not _is_daily_quota(Exception(PER_MINUTE))
    assert _retryable(Exception(PER_MINUTE))
    assert not _retryable(QuotaExhausted(DAILY))


def test_limiter_admits_until_token_budget_then_corrects():
    lim = _Limiter(rpm=30, tpm=8000)
    lim.wait(3000)
    lim.wait(3000)  # 6000 <= 7200 (90% of 8000)
    assert sum(n for _, n in lim.events) == 6000
    lim.correct(3000, 500)
    assert sum(n for _, n in lim.events) == 3500


def test_extract_json_from_prose():
    assert _extract_json('Sure! ```json\n{"a": 1}\n``` done') == '{"a": 1}'
