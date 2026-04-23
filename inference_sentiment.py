from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from prompts import SENTIMENT_LABELS, SENTIMENT_TO_LABEL_ID, SYSTEM_PROMPT, build_user_prompt

DEFAULT_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:8001/v1/chat/completions")
FALLBACK_API_URLS = (
    "http://localhost:8001/v1/chat/completions",
    "http://localhost:8000/v1/chat/completions",
)
DEFAULT_MODEL = "gemma-4-26B-A4B"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 120.0
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.5


class SentimentOutput(BaseModel):
    input_text: str = Field(description="The original Bangla text")
    sentiment: Literal[
        "Strongly Negative",
        "Weakly Negative",
        "Neutral",
        "Weakly Positive",
        "Strongly Positive",
    ] = Field(description="Predicted sentiment label")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    label_id: int = Field(ge=0, le=4, description="Numeric label ID")


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def build_chat_payload(
    text: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = JSON_BLOCK_RE.search(stripped)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in model response.")


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
    repaired = repaired.replace("\u00a0", " ")
    repaired = TRAILING_COMMA_RE.sub(r"\1", repaired)
    return repaired


def normalize_payload(payload: dict[str, Any], original_text: str) -> dict[str, Any]:
    if "input_text" not in payload:
        payload["input_text"] = original_text

    sentiment = payload.get("sentiment")
    label_id = payload.get("label_id")

    if sentiment in SENTIMENT_TO_LABEL_ID and label_id is None:
        payload["label_id"] = SENTIMENT_TO_LABEL_ID[sentiment]
    elif isinstance(label_id, int) and sentiment is None and 0 <= label_id < len(SENTIMENT_LABELS):
        payload["sentiment"] = SENTIMENT_LABELS[label_id]
    elif sentiment in SENTIMENT_TO_LABEL_ID and isinstance(label_id, int):
        payload["label_id"] = SENTIMENT_TO_LABEL_ID[sentiment]

    return payload


def parse_sentiment_response(raw_content: str, original_text: str) -> SentimentOutput:
    last_error: Exception | None = None
    candidates = [raw_content]

    try:
        candidates.append(extract_json_text(raw_content))
    except ValueError as exc:
        last_error = exc

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        for variant in (candidate, repair_json_text(candidate)):
            try:
                payload = json.loads(variant)
                payload = normalize_payload(payload, original_text)
                result = SentimentOutput.model_validate(payload)
                if result.input_text != original_text:
                    result = result.model_copy(update={"input_text": original_text})
                return result
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc

    raise ValueError(f"Failed to parse sentiment JSON: {last_error}") from last_error


def read_input_lines(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text") or record.get("input_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each JSONL line must contain a non-empty 'text' or 'input_text' field.")
            lines.append(text.strip())
        return lines

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_output_records(path: Path, results: list[SentimentOutput]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json(ensure_ascii=False) + "\n")


def classify_text(
    text: str,
    *,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    client: httpx.Client | None = None,
) -> SentimentOutput:
    payload = build_chat_payload(text, model, temperature, top_p, top_k, max_tokens)
    last_error: Exception | None = None
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    attempted_urls: list[str] = []
    urls_to_try: list[str] = [api_url]

    if api_url == DEFAULT_API_URL:
        for fallback_url in FALLBACK_API_URLS:
            if fallback_url not in urls_to_try:
                urls_to_try.append(fallback_url)

    try:
        for url in urls_to_try:
            attempted_urls.append(url)
            for attempt in range(1, retries + 1):
                try:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    return parse_sentiment_response(content, text)
                except (httpx.HTTPError, KeyError, IndexError, ValueError, ValidationError) as exc:
                    last_error = exc
                    is_last_attempt = attempt == retries
                    should_try_next_url = is_last_attempt and isinstance(exc, httpx.ConnectError)
                    if should_try_next_url:
                        break
                    if is_last_attempt:
                        raise
                    time.sleep(retry_delay * attempt)
    finally:
        if owns_client:
            client.close()

    attempted_display = ", ".join(attempted_urls)
    raise RuntimeError(
        "Sentiment inference failed. "
        f"Tried endpoint(s): {attempted_display}. "
        "Make sure llama.cpp is running and that the OpenAI-compatible server port matches `--api_url` "
        "or the `LLAMA_API_URL` environment variable. "
        f"Last error: {last_error}"
    ) from last_error


def interactive_mode(args: argparse.Namespace) -> int:
    print("Bangla Sentiment Analysis Interactive Mode")
    print("Type Bangla text and press Enter. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            text = input("\nText> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return 0

        try:
            result = classify_text(
                text,
                api_url=args.api_url,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            print(result.model_dump_json(indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bangla sentiment analysis via llama.cpp OpenAI-compatible API.")
    parser.add_argument("--text", help="Single Bangla text to classify.")
    parser.add_argument("--input_file", help="Path to a .txt or .jsonl file for batch classification.")
    parser.add_argument("--output_file", help="Where to write batch results as JSONL.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive CLI mode.")
    parser.add_argument("--api_url", default=DEFAULT_API_URL, help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry_delay", type=float, default=DEFAULT_RETRY_DELAY)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    provided_modes = sum(bool(value) for value in (args.text, args.input_file, args.interactive))
    if provided_modes != 1:
        parser.error("Choose exactly one mode: --text, --input_file, or --interactive.")

    if args.text:
        try:
            result = classify_text(
                args.text,
                api_url=args.api_url,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            print(result.model_dump_json(indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.input_file:
        if not args.output_file:
            parser.error("--output_file is required when using --input_file.")

        input_path = Path(args.input_file)
        output_path = Path(args.output_file)
        texts = read_input_lines(input_path)
        results: list[SentimentOutput] = []

        with httpx.Client(timeout=args.timeout) as client:
            for idx, text in enumerate(texts, start=1):
                try:
                    result = classify_text(
                        text,
                        api_url=args.api_url,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                        retry_delay=args.retry_delay,
                        client=client,
                    )
                    results.append(result)
                    print(f"[{idx}/{len(texts)}] {result.sentiment}")
                except Exception as exc:
                    raise RuntimeError(f"Batch inference failed for item {idx}: {exc}") from exc

        write_output_records(output_path, results)
        print(f"Saved {len(results)} results to {output_path}")
        return 0

    return interactive_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
