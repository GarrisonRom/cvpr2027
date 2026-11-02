"""Explicit model adapter registry; no architecture is selected by filename."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from ..common.errors import ContractError
from .contracts import ModelAdapter


AdapterFactory = Callable[..., ModelAdapter]


@dataclass(frozen=True)
class AdapterRegistration:
    adapter_id: str
    model_ids: tuple[str, ...]
    factory: AdapterFactory


class ModelRegistry:
    """Maps manifest-declared adapter ids to explicit factories."""

    def __init__(self, registrations: Mapping[str, AdapterRegistration] | None = None) -> None:
        self._registrations = dict(registrations or {})

    def register(self, registration: AdapterRegistration) -> None:
        if registration.adapter_id in self._registrations:
            raise ContractError(f"adapter already registered: {registration.adapter_id}")
        if not registration.model_ids:
            raise ContractError("an adapter registration needs at least one model id")
        self._registrations[registration.adapter_id] = registration

    def create(self, *, adapter_id: str, model_id: str, **kwargs: object) -> ModelAdapter:
        try:
            registration = self._registrations[adapter_id]
        except KeyError as exc:
            raise ContractError(f"no adapter registered for {adapter_id}") from exc
        if model_id not in registration.model_ids:
            raise ContractError(
                f"adapter {adapter_id} is not declared for model {model_id}"
            )
        return registration.factory(model_id=model_id, **kwargs)

    def registrations(self) -> tuple[AdapterRegistration, ...]:
        return tuple(self._registrations.values())
