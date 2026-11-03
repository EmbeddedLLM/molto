import sys

from loguru import logger
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_HS256_SECRET_BYTES = 32

LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # Config
    molto_log_level_stderr: str = "INFO"
    # External services
    anthropic_api_key: SecretStr = ""
    aws_bearer_token_bedrock: SecretStr = ""
    deepseek_api_key: SecretStr = ""
    fireworks_api_key: SecretStr = ""
    gemini_api_key: SecretStr = ""
    openai_api_key: SecretStr = ""
    openrouter_api_key: SecretStr = ""
    together_ai_api_key: SecretStr = ""

    @property
    def anthropic_api_key_plain(self) -> str:
        return self.anthropic_api_key.get_secret_value()

    @property
    def deepseek_api_key_plain(self) -> str:
        return self.deepseek_api_key.get_secret_value()

    @property
    def gemini_api_key_plain(self) -> str:
        return self.gemini_api_key.get_secret_value()

    @property
    def openai_api_key_plain(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def openrouter_api_key_plain(self) -> str:
        return self.openrouter_api_key.get_secret_value()

    @property
    def together_ai_api_key_plain(self) -> str:
        return self.together_ai_api_key.get_secret_value()


ENV_CONFIG = EnvConfig()


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=ENV_CONFIG.molto_log_level_stderr.upper(),
        format=LOGURU_FORMAT,
    )


setup_logging()
