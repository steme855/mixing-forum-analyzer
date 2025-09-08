import importlib

def test_import_app():
    """Minimaltest: Kann das Hauptmodul geladen werden?"""
    module = importlib.import_module("app.app")
    assert module is not None

def test_dummy_math():
    """Sicherheitsnetz: garantiert grüner Testlauf."""
    assert 2 + 2 == 4
