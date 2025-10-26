import importlib

def test_imports():
    assert importlib.import_module("preset_advisor.core")
