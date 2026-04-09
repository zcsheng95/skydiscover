"""
  VIS Insight Generation evaluator — LLM as a judge.

  Scores an evolved program's insight (using the chart as visual context) across
  four dimensions, each 0-100. The final combined_score is their average normalized
  to [0, 1]. The chart itself is not scored independently.

  The evolved program must export run() -> {"insight": str, "chart_path": str}.

  Requirements:
      pip install openai
      Environment:
          OPENAI_API_KEY (required)
          JUDGE_MODEL (optional, default gpt-4o-mini)
"""

import base64
import hashlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Dict, Union

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "2048"))
JUDGE_TIMEOUT_SECONDS = int(os.environ.get("JUDGE_TIMEOUT_SECONDS", "120"))

DIMENSION_KEYS = [
    "Correctness & Factuality",
    "Specificity & Traceability",
    "Insightfulness & Depth",
    "So-what quality(Actionability | Predictability | Indication)",
]

SYSTEM_PROMPT = """\
You are a strict data visualization judge.

Evaluate one candidate insight against one chart image using only evidence visible
in the chart. Return only a valid JSON object. Do not use markdown or code fences.

Score each dimension as an integer from 0 to 100:
- Correctness & Factuality
- Specificity & Traceability
- Insightfulness & Depth
- So-what quality(Actionability | Predictability | Indication)

Rules:
- Use only the chart as evidence.
- Penalize unsupported claims heavily.
- Reward precise effect sizes, segments, and time windows when visibly supported.
- Keep the evidence concise, under 120 words.
"""

USER_PROMPT_TEMPLATE = """\
Evaluate the candidate insight against the chart.

Candidate insight:
{insight}

Return a JSON object with exactly this structure:
{{
"insight": "<original insight text>",
"scores": {{
    "Correctness & Factuality": <integer 0-100>,
    "Specificity & Traceability": <integer 0-100>,
    "Insightfulness & Depth": <integer 0-100>,
    "So-what quality(Actionability | Predictability | Indication)": <integer 0-100>
}},
"evidence": "<concise evidence-based rationale under 120 words>",
"conclusion": "<one-sentence overall judgment>"
}}
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(timeout=JUDGE_TIMEOUT_SECONDS)
    return _client


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type(chart_path: str) -> str:
    ext = os.path.splitext(chart_path)[1].lstrip(".").lower()
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")


def _coerce_score(value) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(100.0, float(value)))


def _parse_judge_response(raw: str) -> Dict[str, Union[float, str]]:
    result = json.loads(raw)
    raw_scores = result.get("scores", {})

    scores = {key: _coerce_score(raw_scores.get(key, 0)) for key in DIMENSION_KEYS}
    scores["evidence"] = str(result.get("evidence", "")).strip()
    scores["conclusion"] = str(result.get("conclusion", "")).strip()
    return scores


def _judge(insight: str, chart_path: str) -> Dict[str, Union[float, str]]:
    client = _get_client()
    b64 = _encode_image(chart_path)
    data_url = f"data:{_mime_type(chart_path)};base64,{b64}"

    user_text = USER_PROMPT_TEMPLATE.format(insight=insight)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        },
    ]

    last_error = None
    max_output_tokens = JUDGE_MAX_TOKENS

    for attempt in range(2):
        try:
            response = client.responses.create(
                model=JUDGE_MODEL,
                input=messages,
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "json_object"}},
            )

            raw = (response.output_text or "").strip()
            logger.info("Judge raw response (attempt=%d): %s", attempt + 1, raw[:300])

            if not raw:
                raise ValueError("Empty judge response")

            return _parse_judge_response(raw)

        except Exception as e:
            last_error = e
            logger.warning("Judge attempt %d failed: %s", attempt + 1, e)

            if attempt == 0:
                max_output_tokens = max(1200, max_output_tokens * 2)

                try:
                    debug_dump = response.model_dump() if "response" in locals() else None
                    if debug_dump:
                        logger.debug("Judge response dump: %s", json.dumps(debug_dump)[:2000])
                except Exception:
                    pass

    logger.error("Judge failed after retries: %s", last_error)
    return {key: 0.0 for key in DIMENSION_KEYS}


def _run_program(program_path: str, timeout_seconds: int = 120) -> Dict:
    """Run the evolved program in a subprocess and return its run() output."""
    out_fd, out_path = tempfile.mkstemp(suffix=".out")
    os.close(out_fd)

    script = (
        "import sys, os, pickle, traceback\n"
        f"sys.path.insert(0, os.path.dirname({repr(program_path)}))\n"
        "import importlib.util\n"
        "try:\n"
        f"    spec = importlib.util.spec_from_file_location('program', {repr(program_path)})\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "    result = mod.run()\n"
        "    assert isinstance(result, dict) and 'insight' in result and 'chart_path' in result\n"
        f"    with open({repr(out_path)}, 'wb') as f:\n"
        "        pickle.dump({'ok': result}, f)\n"
        "except Exception as e:\n"
        "    traceback.print_exc()\n"
        f"    with open({repr(out_path)}, 'wb') as f:\n"
        "        pickle.dump({'error': str(e)}, f)\n"
    )

    script_fd, script_path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(script_fd, "w") as f:
        f.write(script)

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"Program timed out after {timeout_seconds}s")

        if stdout:
            logger.debug("Program stdout: %s", stdout.decode(errors="replace")[:500])
        if stderr:
            logger.debug("Program stderr: %s", stderr.decode(errors="replace")[:1000])

        if not os.path.exists(out_path):
            raise RuntimeError(f"Program produced no output (exit code {proc.returncode})")

        with open(out_path, "rb") as f:
            payload = pickle.load(f)

        if "error" in payload:
            raise RuntimeError(f"Program raised: {payload['error']}")

        return payload["ok"]

    finally:
        for p in (script_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def evaluate(program_path: str) -> Dict[str, Union[float, str]]:
    """
    Score an evolved insight+chart program using an OpenAI judge model.

    Args:
        program_path: Path to the evolved Python program file.
            The program must export run() -> {"insight": str, "chart_path": str}.

    Returns:
        Dict with combined_score (0-1), per-category scores (0-1), and judge evidence.
    """
    try:
        result = _run_program(program_path, timeout_seconds=120)
    except Exception as e:
        logger.error("Program execution failed: %s", e)
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}

    insight = str(result.get("insight", "")).strip()
    chart_path = str(result.get("chart_path", "")).strip()

    if not insight:
        return {"combined_score": 0.0, "error": "No insight returned"}
    if not chart_path or not os.path.exists(chart_path):
        return {"combined_score": 0.0, "error": f"Chart not found: {chart_path}"}

    with open(program_path, "rb") as _pf:
        _prog_hash = hashlib.sha256(_pf.read()).hexdigest()[:16]
    _charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_charts")
    os.makedirs(_charts_dir, exist_ok=True)
    stable_path = os.path.join(_charts_dir, f"{_prog_hash}.png")
    try:
        shutil.copy2(chart_path, stable_path)
        chart_path = stable_path
    except OSError:
        pass  # fall back to /tmp path

    scores = _judge(insight, chart_path)

    dim_scores = [scores.get(key, 0.0) for key in DIMENSION_KEYS]
    combined = round(sum(dim_scores) / (100.0 * len(DIMENSION_KEYS)), 4)

    out = {
        "combined_score": combined,
        "insight_text": insight,
        "image_path": chart_path,
        "image_path_stable": stable_path,
    }

    for key in DIMENSION_KEYS:
        out[key] = round(scores.get(key, 0.0) / 100.0, 4)

    if scores.get("evidence"):
        out["judge_evidence"] = scores["evidence"]
    if scores.get("conclusion"):
        out["judge_conclusion"] = scores["conclusion"]

    return out