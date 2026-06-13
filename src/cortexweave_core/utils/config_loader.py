# config/config_loader.py
import os
from dotenv import load_dotenv

class ConfigLoader:
    """
    Loads environment variables from .env.<APP_ENV> file.
    Priority (highest → lowest):
        1. Real OS environment variables  (CI/CD, Docker, K8s secrets)
        2. .env.<env> file               (environment-specific)
        3. .env file                     (local overrides, not committed)
    """

    def __init__(self, env: str = None):
        self.env = env or os.getenv("APP_ENV", "dev")
        self.env_dir = os.getenv("ENV_FOLDER", ".")
        self._load_env_files()


    def _load_env_files(self) -> None:
        base_env_path = os.path.join(self.env_dir, ".env")
        specific_env_path = os.path.join(self.env_dir, f".env.{self.env}")

        # Load .env first (base/local overrides)
        if os.path.exists(base_env_path):
            load_dotenv(base_env_path, override=False)

        # Load .env.<env> — higher priority
        if not os.path.exists(specific_env_path):
            abs_path = os.path.abspath(specific_env_path)
            raise FileNotFoundError(
                f"Config file not found!\n"
                f"Expected path: {abs_path}\n"
                f"ENV_FOLDER set to: {self.env_dir}\n"
                f"CWD: {os.getcwd()}"
            )
        # override=True so env-specific file wins over .env
        load_dotenv(specific_env_path, override=True)


    def get(self, key: str, default=None):
        """
        config.get("repository.s3.bucket")
        config.get("logging.level", "INFO")
        """
        value = os.getenv(key)
        if value is None:
            return default
        return self._cast(value)

    def _cast(self, value: str):
        """Auto-cast booleans and integers from string env values."""
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    def __repr__(self):
        return f"ConfigLoader(env={self.env})"

config = ConfigLoader()
