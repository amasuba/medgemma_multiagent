"""
message_queue.py
MedGemma Multi-AI Agentic System

Inter-agent communication system enabling coordinated workflows.
Author: Aaron Masuba
License: MIT
"""

import asyncio
from typing import Any, Dict, Optional, Callable
from loguru import logger

class Message:
    """
    Encapsulates a message exchanged between agents.
    """
    def __init__(self, sender: str, recipient: str, payload: Dict[str, Any]):
        self.sender = sender
        self.recipient = recipient
        self.payload = payload

class MessageQueue:
    """
    Asynchronous, in-memory message queue for inter-agent communication.
    Supports publish/subscribe and request/reply patterns.
    """
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._logger = logger.bind(component="MessageQueue")

    def register_agent(self, agent_name: str) -> None:
        """
        Create a dedicated queue for an agent if not already present.
        """
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()
            self._logger.debug(f"Registered agent queue: {agent_name}")

    async def send(
        self,
        recipient: str,
        message: Message,
        timeout: Optional[float] = None
    ) -> None:
        """
        Send a message to the recipient's queue.
        """
        if recipient not in self._queues:
            raise KeyError(f"Recipient queue not found: {recipient}")
        try:
            await asyncio.wait_for(self._queues[recipient].put(message), timeout=timeout)
            self._logger.debug(f"Sent message from {message.sender} to {recipient}")
        except asyncio.TimeoutError:
            self._logger.error(f"Timeout sending message to {recipient}")

    async def receive(
        self,
        agent_name: str,
        timeout: Optional[float] = None
    ) -> Message:
        """
        Await a message for the given agent.
        """
        if agent_name not in self._queues:
            raise KeyError(f"Agent queue not found: {agent_name}")
        try:
            msg = await asyncio.wait_for(self._queues[agent_name].get(), timeout=timeout)
            self._logger.debug(f"{agent_name} received message from {msg.sender}")
            return msg
        except asyncio.TimeoutError as e:
            self._logger.error(f"Timeout waiting for message in {agent_name}")
            raise e

    async def broadcast(
        self,
        message: Message,
        timeout: Optional[float] = None
    ) -> None:
        """
        Send the same message to all registered agents.
        """
        for agent in self._queues:
            await self.send(agent, message, timeout=timeout)

    def queue_size(self, agent_name: str) -> int:
        """
        Return the current size of the agent's queue.
        """
        if agent_name not in self._queues:
            raise KeyError(f"Agent queue not found: {agent_name}")
        return self._queues[agent_name].qsize()

    async def clear(self) -> None:
        """
        Empty all queues.
        """
        for name, q in self._queues.items():
            drained = 0
            while not q.empty():
                _ = q.get_nowait()
                drained += 1
            self._logger.debug(f"Cleared {drained} messages from queue '{name}'")
