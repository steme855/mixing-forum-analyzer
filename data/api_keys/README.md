# API-Key-Daten

Dieser Ordner ist fuer die Monetarisierungsgrundlage der API vorgesehen.

- `keys.json`: gehashte API-Key-Metadaten, Quotas und Plan-Informationen
- `usage.jsonl`: append-only Nutzungslog
- Schreibpfad: `src/api_access/APIKeyStore`

Klartext-API-Keys werden nicht gespeichert. Runtime-Dateien werden nicht versioniert.
