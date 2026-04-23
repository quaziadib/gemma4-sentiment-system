import asyncio
import re
import httpx
from config import settings
from models import TranslationRequest, TranslationResponse


TRANSLATION_PROMPT = (
    "Translate the text below from {source_language} to {target_language}.\n"
    "Rules:\n"
    "- Output the translation only. Nothing else.\n"
    "- Do not include any tags, labels, commentary, or preamble.\n"
    "- Do not repeat or explain the instruction.\n\n"
    "{text}"
)

# Matches any angle-bracket tag or pipe-delimited special token,
# including unclosed ones like <channelFoo (no '>').
_TAG_RE = re.compile(r"<[^>]{0,80}>?|<\|[^|>]*\|?>?|\|>", re.IGNORECASE)

# Leading-line artifacts the model emits before the actual translation.
_ARTIFACT_LINE_RE = re.compile(
    r"^\s*(thought|think|channel|translation|output|answer|result)\s*$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    # Remove all tag-like tokens
    text = _TAG_RE.sub("", text)
    # Drop leading lines that are bare artifact words with no real content
    lines = text.splitlines()
    while lines and _ARTIFACT_LINE_RE.match(lines[0]):
        lines.pop(0)
    text = "\n".join(lines)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(req: TranslationRequest) -> str:
    return TRANSLATION_PROMPT.format(
        source_language=req.source_language,
        target_language=req.target_language,
        text=req.text,
    )


def build_chat_payload(req: TranslationRequest) -> dict:
    return {
        "model": settings.vllm_model_name,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(req),
            }
        ],
        "max_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "repetition_penalty": settings.repetition_penalty,
    }


class TranslatorService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self._url = f"{settings.vllm_base_url}/v1/chat/completions"

    async def translate(self, req: TranslationRequest) -> TranslationResponse:
        payload = build_chat_payload(req)
        response = await self._client.post(
            self._url,
            json=payload,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        translated = _clean(data["choices"][0]["message"]["content"])
        return TranslationResponse(
            original_text=req.text,
            translated_text=translated,
            source_language=req.source_language,
            target_language=req.target_language,
        )

    async def translate_batch(
        self, requests: list[TranslationRequest]
    ) -> list[TranslationResponse]:
        tasks = [self.translate(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def is_healthy(self) -> bool:
        try:
            resp = await self._client.get(
                f"{settings.vllm_base_url}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
