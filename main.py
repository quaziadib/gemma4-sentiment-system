import asyncio
import logging
import time

import httpx
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from inference_sentiment import DEFAULT_API_URL, SentimentOutput, classify_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("sentiment-api")

API_MAX_TOKENS = 256


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class BatchSentimentRequest(BaseModel):
    requests: list[SentimentRequest] = Field(..., min_length=1, max_length=32)


class BatchSentimentResponse(BaseModel):
    results: list[SentimentOutput]


class HealthResponse(BaseModel):
    status: str
    vllm_reachable: bool
    model: str
    inference_url: str

# Shared state
_http_client: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, _semaphore
    limits = httpx.Limits(
        max_connections=settings.max_concurrent_requests,
        max_keepalive_connections=settings.max_concurrent_requests // 2,
    )
    _http_client = httpx.AsyncClient(
        limits=limits,
        headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
    )
    _semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    logger.info("HTTP client pool initialised (max=%d)", settings.max_concurrent_requests)
    yield
    await _http_client.aclose()
    logger.info("HTTP client pool closed")


app = FastAPI(
    title="Gemma 4 Bangla Sentiment Analysis API",
    description="FastAPI backend for Bangla 5-class sentiment analysis powered by llama.cpp + Gemma 4",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_inference_url() -> str:
    return f"{settings.vllm_base_url}/v1/chat/completions" if settings.vllm_base_url else DEFAULT_API_URL


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    if _http_client is None:
        raise RuntimeError("HTTP client not initialised")

    inference_url = get_inference_url()
    health_url = inference_url.removesuffix("/v1/chat/completions") + "/health"

    try:
        response = await _http_client.get(health_url, timeout=5.0)
        reachable = response.status_code == 200
    except Exception:
        reachable = False

    return HealthResponse(
        status="ok" if reachable else "degraded",
        vllm_reachable=reachable,
        model=settings.vllm_model_name,
        inference_url=inference_url,
    )


@app.post(
    "/sentiment",
    response_model=SentimentOutput,
    status_code=status.HTTP_200_OK,
    tags=["sentiment"],
)
async def analyze_sentiment(req: SentimentRequest):
    async with _semaphore:
        try:
            return await asyncio.to_thread(
                classify_text,
                req.text,
                api_url=get_inference_url(),
                model=settings.vllm_model_name,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=API_MAX_TOKENS,
                timeout=settings.request_timeout,
            )
        except httpx.HTTPStatusError as exc:
            logger.error("Sentiment backend error: %s", exc.response.text)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Sentiment backend returned an error.",
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Sentiment backend timed out.",
            )
        except Exception as exc:
            logger.exception("Unexpected error during sentiment inference")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )


@app.post(
    "/sentiment/batch",
    response_model=BatchSentimentResponse,
    status_code=status.HTTP_200_OK,
    tags=["sentiment"],
)
async def analyze_sentiment_batch(batch: BatchSentimentRequest):
    async def run_single(item: SentimentRequest) -> SentimentOutput:
        async with _semaphore:
            return await asyncio.to_thread(
                classify_text,
                item.text,
                api_url=get_inference_url(),
                model=settings.vllm_model_name,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=API_MAX_TOKENS,
                timeout=settings.request_timeout,
            )

    try:
        results = await asyncio.gather(*(run_single(item) for item in batch.requests))
        return BatchSentimentResponse(results=results)
    except httpx.HTTPStatusError as exc:
        logger.error("Sentiment batch backend error: %s", exc.response.text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Sentiment backend returned an error.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Sentiment backend timed out.",
        )
    except Exception as exc:
        logger.exception("Unexpected error during batch sentiment inference")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level="info",
    )
