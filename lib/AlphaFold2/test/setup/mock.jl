# we need to mock the kernels used in openfold:
py"""
import sys
from types import ModuleType
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec

class MockModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        # Setting __path__ tells Python this is a package container,
        # allowing seamless dot-notation sub-imports (e.g., deepspeed.ops)
        self.__path__ = [] 

    def __getattr__(self, attr):
        # Guard internal double-underscore attributes.
        # Allowing Python to handle these naturally prevents infinite recursion loops.
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)
        # Return a MagicMock for any standard method, class, or variable call
        return MagicMock()

class MockLoader:
    def create_module(self, spec):
        # Dynamically instantiate our custom module type
        return MockModule(spec.name)
        
    def exec_module(self, module):
        # No-op. We don't have code to execute, but defining this method
        # makes this a fully compliant Python loader.
        pass 

class MockPackage:
    def __init__(self, blocked_prefixes):
        self.blocked_prefixes = blocked_prefixes
        self.loader = MockLoader()

    def find_spec(self, fullname, path, target=None):
        base_module = fullname.split('.')[0]
        if base_module in self.blocked_prefixes:
            # Return a perfectly formed ModuleSpec. Because we provide a valid loader,
            # Python's core engine will automatically generate and attach a real,
            # non-None '__spec__' object to the module for us.
            return ModuleSpec(fullname, self.loader, is_package=True)
        return None

# 1. Clear out any broken or contaminated mocks from previous runs
for mod in list(sys.modules.keys()):
    if mod.split('.')[0] in {'deepspeed', 'flash_attn', 'attn_core_inplace_cuda'}:
        del sys.modules[mod]

# 2. Register our blocker at the absolute front of Python's import line
sys.meta_path.insert(0, MockPackage({'deepspeed', 'flash_attn', 'attn_core_inplace_cuda'}))
"""