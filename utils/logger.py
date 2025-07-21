"""
logger.py
MedGemma Multi-AI Agentic System

Comprehensive logging setup with structured loguru-based loggers,
performance tracking, and per-agent contextual binding.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger
import sys

class LoggerConfig:
    """
    Configuration for the logging system.
    Reads from environment variables or defaults.
    """
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR: Optional[Path] = Path(os.getenv("LOG_DIR", "./logs"))
    FILE_MAX_SIZE: str = os.getenv("FILE_MAX_SIZE", "10MB")
    BACKUP_COUNT: int = int(os.getenv("BACKUP_COUNT", "5"))
    STRUCTURED: bool = os.getenv("STRUCTURED_LOGS", "true").lower() == "true"
    PERFORMANCE_ENABLED: bool = os.getenv("PERFORMANCE_LOGGING", "true").lower() == "true"
    PERFORMANCE_INTERVAL: int = int(os.getenv("PERFORMANCE_INTERVAL", "60"))

def setup_logger(name: str, log_level: Optional[str] = None, log_dir: Optional[str] = None):
    """
    Initialize structured logging for the system.

    Args:
        name: Name to bind to all log entries (e.g., orchestrator or agent name).
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory path to write rotating log files.
    """
    cfg = LoggerConfig()
    level = (log_level or cfg.LOG_LEVEL).upper()

    # Remove existing handlers to avoid duplication
    logger.remove()

    # Console sink
    logger.add(
        sink=sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )

    # File sink (rotating)
    if log_dir or cfg.LOG_DIR:
        log_path = Path(log_dir or cfg.LOG_DIR)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=str(log_path / f"{name}.log"),
            level=level,
            rotation=cfg.FILE_MAX_SIZE,
            retention=cfg.BACKUP_COUNT,
            serialize=cfg.STRUCTURED,
            enqueue=True,
        )

    # Performance tracking
    if cfg.PERFORMANCE_ENABLED:
        # Periodically log memory and CPU usage every PERFORMANCE_INTERVAL seconds
        def _log_performance():
            import psutil, threading
            proc = psutil.Process(os.getpid())

            def _worker():
                mem = proc.memory_info().rss / (1024 * 1024)
                cpu = proc.cpu_percent(interval=None)
                logger.bind(component=name).debug(
                    f"PERF | memory={mem:.1f}MB | cpu={cpu:.1f}%"
                )
                threading.Timer(cfg.PERFORMANCE_INTERVAL, _worker).start()

            _worker()

        try:
            _log_performance()
        except Exception:
            logger.warning("Performance tracking failed to start")

    return logger.bind(component=name)
