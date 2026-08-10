from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="molto_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # Config
    log_level_stderr: str = "INFO"
    # External services
    anthropic_api_key: SecretStr = ""
    deepseek_api_key: SecretStr = ""
    gemini_api_key: SecretStr = ""
    openai_api_key: SecretStr = ""

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


ENV_CONFIG = EnvConfig()
