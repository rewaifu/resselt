from .registry import Registry
from .architecture import Architecture, WrappedModel, KeyCondition

global_registry = Registry()

__all__ = ["global_registry", "Architecture", "WrappedModel", "KeyCondition"]
