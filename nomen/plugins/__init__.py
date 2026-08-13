"""Plugin system for extending generators / validators.

Plugins run in the orchestrator process only. They are not pickled into
ProcessPoolExecutor workers — register them before `run_engine`, and do not
expect generator plugins to be chunked like the built-in engines.
"""

from __future__ import annotations

from typing import Protocol

from nomen.models import Candidate


class GeneratorPlugin(Protocol):
    name: str

    def generate(self, count: int) -> list[str]: ...


class ValidatorPlugin(Protocol):
    name: str

    async def validate(self, candidate: Candidate) -> list[str]:
        """Return rejection reasons (empty if ok)."""
        ...


_GENERATORS: list[GeneratorPlugin] = []
_VALIDATORS: list[ValidatorPlugin] = []


def register_generator(plugin: GeneratorPlugin) -> None:
    _GENERATORS.append(plugin)


def register_validator(plugin: ValidatorPlugin) -> None:
    _VALIDATORS.append(plugin)


def iter_generators() -> list[GeneratorPlugin]:
    return list(_GENERATORS)


def iter_validators() -> list[ValidatorPlugin]:
    return list(_VALIDATORS)
