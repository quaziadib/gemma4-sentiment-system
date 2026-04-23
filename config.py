from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # VLLM
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "gemma-4-26B-A4B-it-UD-Q4_K_XL"
    vllm_api_key: str = "none"

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 9000
    api_workers: int = 4
    request_timeout: float = 120.0

    # Concurrency
    max_concurrent_requests: int = 64

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
