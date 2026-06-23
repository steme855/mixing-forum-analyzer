# Changelog

## [Unreleased]

### Added
- Documented recent fixes and README translation in this changelog.

### Changed
- Translated the project README to English for international contributors.
- Semantic search now preserves user-provided corpora without silently falling back to defaults and keeps custom document IDs intact.
- Runtime dependencies are now pinned (including `sentence-transformers`, `torch`, `spacy`, and `de-core-news-sm`) and repository data paths resolve via absolute `Path` objects for deployment safety.

### Fixed
- Resolved an issue where passing an empty corpus defaulted to bundled demo data and where search results ignored explicit document identifiers.
- Hardened spaCy initialisation with `@st.cache_resource` and an automatic `de_core_news_sm` download fallback to stop startup failures on fresh environments.
