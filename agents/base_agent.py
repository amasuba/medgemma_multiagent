"""
base_agent.py
MedGemma Multi-AI Agentic System

Author: Aaron Masuba
License: MIT
"""

from __future__ import annotations

import abc
import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

# --------------------------------------------------------------------------- #
# Helper exceptions
# --------------------------------------------------------------------------- #


class AgentNotInitializedError(RuntimeError):
    """Raised when an agent is used before initialize() is called."""


class AgentTimeoutError(RuntimeError):
    """Raised when the agent exceeds its allotted processing time."""


# --------------------------------------------------------------------------- #
# BaseAgent definition
# --------------------------------------------------------------------------- #


class BaseAgent(abc.ABC):
    """
    Abstract superclass for all MedGemma agents.

    Sub-classes MUST implement:
        - _initialize()
        - _process()
        - _shutdown()
    """

    # ------ Public API ------------------------------------------------------ #

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        global_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg : Dict[str, Any]
            Agent-specific configuration node loaded from config.yaml.
        global_cfg : Dict[str, Any] | None
            Optional global settings (timeouts, retries, etc.).
        """
        self.cfg = cfg or {}
        self.global_cfg = global_cfg or {}

        self.name: str = self.cfg.get("name", self.__class__.__name__)
        self.description: str = self.cfg.get(
            "description", f"{self.name} for MedGemma workflow"
        )

        # Operational parameters ------------------------------------------------
        self._timeout: int = int(
            self.global_cfg.get("timeout", 300)
        )  # seconds
        self._retry_attempts: int = int(
            self.global_cfg.get("retry_attempts", 3)
        )
        self._communication_timeout: int = int(
            self.global_cfg.get("communication_timeout", 60)
        )

        # Lifecycle flags -------------------------------------------------------
        self._initialized: bool = False
        self._shutdown_event = asyncio.Event()

        # Structured logger -----------------------------------------------------
        level = self.global_cfg.get("log_level", "INFO").upper()
        logger.remove()  # Avoid duplicate handlers if multiple agents
        logger.add(
            sink="sys.stderr",
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>{self.name}</cyan> | "
            "<level>{message}</level>",
            backtrace=False,
            diagnose=False,
        )
        self.log = logger.bind(agent=self.name.lower())

    # --------------------------------------------------------------------- #
    # Async lifecycle helpers
    # --------------------------------------------------------------------- #

    async def initialize(self) -> None:
        """Initialize resources (e.g., model weights, DB connections)."""
        if self._initialized:
            self.log.warning("initialize() called twice — skipping")
            return

        start = time.perf_counter()
        self.log.debug("Initializing …")

        await self._initialize()

        elapsed = (time.perf_counter() - start) * 1,000
        self.log.info(f"Initialized in {elapsed:.2f} ms")
        self._initialized = True

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entrypoint for performing the agent’s core task.

        Implements retries, timeouts, and exception safety.
        """
        if not self._initialized:
            raise AgentNotInitializedError(
                f"{self.name} used before initialize()"
            )

        attempt = 0
        while attempt < self._retry_attempts:
            attempt += 1
            try:
                self.log.debug(
                    f"Starting _process() (attempt {attempt}/{self._retry_attempts})"
                )
                return await asyncio.wait_for(
                    self._process(payload), timeout=self._timeout
                )
            except asyncio.TimeoutError as exc:
                self.log.error(f"Timeout after {self._timeout}s")
                if attempt >= self._retry_attempts:
                    raise AgentTimeoutError(
                        f"{self.name} timed-out after {self._retry_attempts} attempts"
                    ) from exc
            except Exception as exc:  # noqa: BLE001
                self.log.exception(f"Processing error: {exc}")
                if attempt >= self._retry_attempts:
                    raise
                await asyncio.sleep(1.0)  # linear back-off
        # Should never reach here
        return {}  # type: ignore[return-value]

    async def shutdown(self) -> None:
        """Gracefully release resources before exit."""
        if not self._initialized:
            return
        self.log.debug("Shutting down …")
        await self._shutdown()
        self._shutdown_event.set()
        self.log.info("Shutdown complete ✔")

    # --------------------------------------------------------------------- #
    # Abstract methods (must be provided by subclasses)
    # --------------------------------------------------------------------- #

    @abc.abstractmethod
    async def _initialize(self) -> None:  # pragma: no cover
        """Agent-specific initialization steps."""
        raise NotImplementedError

    @abc.abstractmethod
    async def _process(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        """Agent-specific task implementation."""
        raise NotImplementedError

    @abc.abstractmethod
    async def _shutdown(self) -> None:  # pragma: no cover
        """Agent-specific teardown steps."""
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #

    def is_shutdown(self) -> bool:
        """Return True once shutdown() completes."""
        return self._shutdown_event.is_set()

    async def wait_until_shutdown(self, timeout: Optional[int] = None) -> None:
        """Block until shutdown completes (useful for orchestrators)."""
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(), timeout=timeout
            )
        except asyncio.TimeoutError:
            self.log.warning("Shutdown wait timed-out")
