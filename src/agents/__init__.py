"""Collects all defined and implemented agent models."""

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent

__all__ = ["BaseAgent", "DQNAgent"]
