#!/usr/bin/env python3
"""
Check Python Version und ML-Library Kompatibilität
"""

import sys
import platform

def check_python_ml_compatibility():
    """Überprüft Python-Version und ML-Kompatibilität"""

    print("🐍 PYTHON & ML-KOMPATIBILITÄT CHECK")
    print("=" * 50)

    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")

    py_version = sys.version_info

    print("\n📊 KOMPATIBILITÄTS-BEWERTUNG:")

    if py_version.major == 3 and py_version.minor == 12:
        print("🔴 Python 3.12: PROBLEMATISCH für ML-Libraries")
        print("   • spaCy: Bekannte Import-Probleme")
        print("   • sentence-transformers: Hängt oft beim Import")  
        print("   • transformers: Sehr langsam beim ersten Import")
        print("\n💡 EMPFEHLUNG: Wechsel zu Python 3.11 oder 3.10")

    elif py_version.major == 3 and py_version.minor == 11:
        print("✅ Python 3.11: OPTIMAL für ML-Libraries")
        print("   • Beste Balance aus Performance und Kompatibilität")

    elif py_version.major == 3 and py_version.minor == 10:
        print("✅ Python 3.10: SEHR GUT für ML-Libraries")
        print("   • Bewährt und stabil mit allen ML-Frameworks")

    elif py_version.minor < 10:
        print("⚠️  Python < 3.10: Teilweise veraltet")
        print("   • Manche neuere Features nicht verfügbar")

    else:
        print(f"❓ Python {py_version.major}.{py_version.minor}: Unbekannt")

    # Teste verfügbare Alternativen
    print("\n🔄 PYTHON-VERSION WECHSELN:")
    print("# Neue venv mit Python 3.11 erstellen:")
    print("python3.11 -m venv venv_311")
    print("source venv_311/bin/activate")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    check_python_ml_compatibility()
