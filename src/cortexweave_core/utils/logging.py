# core/logging.py
import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone

from .config_loader import config


class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                log_entry[key] = value
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class PlainFormatter(logging.Formatter):
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def setup_logging() -> None:
    level = getattr(logging, config.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_format = config.get("LOG_FORMAT", "plain")
    output = config.get("LOG_OUTPUT", "console")

    formatter = JsonFormatter() if log_format == "json" else PlainFormatter()
    handlers = []

    if output in ("console", "both"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if output in ("file", "both"):
        log_path = config.get("LOG_FILE_PATH", "logs/app.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=config.get("LOG_FILE_MAX_BYTES", 10_485_760),
            backupCount=config.get("LOG_FILE_BACKUP_COUNT", 5),
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised",
        extra={"env": config.env, "level": logging.getLevelName(level), "format": log_format}
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
