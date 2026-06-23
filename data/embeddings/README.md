# Embedding Outputs

This directory is reserved for generated vector artifacts.

Expected runtime outputs:

- TF-IDF vectorizer/cache files
- SBERT embedding matrices
- versioned embedding snapshots such as `v1/`

Generated embedding files are intentionally not part of the seed dataset. They
depend on model versions, local environment, and corpus version.

Recommended metadata for future generated artifacts:

```json
{
  "corpus_id": "mfa_seed_mixing_forum_corpus",
  "corpus_version": "1.0",
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "document_count": 32
}
```
