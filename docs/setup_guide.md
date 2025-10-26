# CI/CD & Deployment Setup Guide

Dieses Dokument führt dich in ~20 Minuten durch die komplette Integration der neuen Tooling- und Deployment-Pipeline.

## 1. Vorbereitung (5 Minuten)
1. Python 3.11 lokal installieren und `python -m pip install --upgrade pip` ausführen.
2. Repository clonen und virtuelle Umgebung anlegen.
3. Entwicklungs-Abhängigkeiten installieren:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

## 2. GitHub Actions konfigurieren (5 Minuten)
1. Öffne die Repository-Settings → Secrets and variables → Actions.
2. Lege folgende Secrets an:
   - `CODECOV_TOKEN`: Token aus https://app.codecov.io
   - `HF_TOKEN`: Hugging Face Access Token mit Schreibrechten
   - `HF_SPACE_ID`: z.B. `username/mixing-forum-analyzer`
3. Optional: `HF_HEALTH_URL`, falls der Health-Check auf ein anderes Endpoint zeigen soll.

## 3. Hugging Face Deployment (5 Minuten)
1. Stelle sicher, dass auf Hugging Face bereits eine Space mit Streamlit-Template existiert.
2. Setze unter *Settings → Integrations* den Git-Mode auf “Write access” für CI.
3. Prüfe, ob `requirements.txt` und `app/` alle nötigen Assets enthalten.
4. Der Deploy-Workflow triggert automatisch bei jedem `main`-Push, sofern `app/`, `requirements.txt` oder `README.md` verändert wurden.

## 4. Tests & Qualitäts-Gates (3 Minuten)
- `pytest.ini` konfiguriert Coverage, Timeout (30 s) und Marker (`unit`, `integration`, `slow`).
- `pytest --maxfail=1` liefert schnelle Rückmeldung im lokalen TDD-Zyklus.
- `make lint` (flake8, black, isort) und `make test` sind die Minimal-Vorbereitung für einen Pull Request.

## 5. Best Practices (TDD-Loop) (2 Minuten)
1. Schreibe zuerst einen Test (`pytest -k name -x`).
2. Implementiere minimale Funktionalität.
3. Refaktoriere mit `make format`.
4. Wiederhole, bis CI (`Tests`-Workflow) grün ist.

## 6. Troubleshooting
| Problem | Lösung |
|---------|--------|
| Workflow stoppt beim Hugging Face Upload | Prüfe, ob `HF_TOKEN` & `HF_SPACE_ID` gesetzt sind und der Token Schreibrechte hat. |
| Codecov Schritt schlägt fehl | Token neu generieren, `CODECOV_TOKEN` aktualisieren, Coverage-Datei (`coverage.xml`) existiert? |
| Black/Flake8 schlagen lokal fehl | `make format` ausführen, anschließend `make lint`.
| Streamlit Deployment zeigt alte Version | Space auf Hugging Face “Restart” oder Workflow erneut triggern (leer commit).

## 7. README Badge Integration
Füge im oberen Bereich des README folgende Badges ein (Owner anpassen):
```markdown
[![Tests](https://github.com/<owner>/mixing-forum-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/<owner>/mixing-forum-analyzer/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/<owner>/mixing-forum-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/<owner>/mixing-forum-analyzer)
```

Damit steht ein CI/CD-System mit Enterprise-Anspruch bereit: Linting, Formatierung, Tests, Coverage und automatisches Deployment inklusive Health-Checks.
