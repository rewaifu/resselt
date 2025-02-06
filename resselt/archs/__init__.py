import importlib.util
import os

from ..factory import Architecture
from ..registry import Registry

internal_registry = Registry()
base_dir = os.path.dirname(__file__)
base_class = Architecture

for root, dirs, files in os.walk(base_dir):
    if root == base_dir:
        continue

    for file in files:
        if file.endswith('.py'):
            rel_dir = os.path.relpath(root, base_dir)
            module_name = os.path.join(rel_dir, file[:-3]).replace(os.path.sep, '.')
            full_module_name = f'{__name__}.{module_name}'
            module = importlib.import_module(full_module_name)

            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, base_class) and attribute is not base_class:
                    globals()[attribute_name] = attribute

                    instance = attribute()
                    internal_registry.add(instance)
