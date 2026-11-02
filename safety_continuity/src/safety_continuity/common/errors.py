"""Errors shared by contract validation and experiment entry points."""


class ContractError(ValueError):
    """Raised when a cross-module experiment contract is incomplete or incompatible."""


class ConfigurationError(ContractError):
    """Raised when a YAML configuration cannot be resolved safely."""
