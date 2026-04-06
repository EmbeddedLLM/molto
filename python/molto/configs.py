from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_HS256_SECRET_BYTES = 32


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="molto_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # Config
    log_dir: str = "logs"
    log_level_stderr: str = "INFO"
    log_level_file: str = "INFO"
    log_timings: bool = False
    # External services
    anthropic_api_key: SecretStr = ""
    gemini_api_key: SecretStr = ""
    openai_api_key: SecretStr = ""
    openrouter_api_key: SecretStr = ""
    together_ai_api_key: SecretStr = ""

    @property
    def anthropic_api_key_plain(self) -> str:
        return self.anthropic_api_key.get_secret_value()

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
