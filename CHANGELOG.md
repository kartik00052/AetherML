# Changelog

All notable changes to AetherML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added
- Initial release of AetherML SDK
- Multi-agent pipeline architecture with LangGraph orchestration
- Data engine abstraction (pandas, polars; Spark stubbed)
- Agents: upload, ETL, validation, EDA, target detection, feature engineering, model selection, evaluation, explainability, reporting
- AutoML: rule-based model recommendation, GridSearchCV training, metric evaluation
- SHAP-based model explainability with fallback chain
- RAG infrastructure (Qdrant, sentence-transformers)
- Typer CLI interface
- Pydantic-based configuration system
- Comprehensive test suite for implemented agents and ML modules

### Changed
- Simplified ReportingAgent — removed LLM narrative generation, now focuses on structured report assembly
- Reduced codebase from 96 to 89 source files via LLM module removal
- Improved type safety — mypy passes with 0 errors across all source files
- Improved performance — eliminated redundant DataFrame copies, engine.collect() calls, and cached RAG clients

### Removed
- LLM module (`aetherml/llm/`) — GemmaClient, prompt builder, response parser
- `LLMConfig` from settings, `LLMError`/`LLMTimeoutError`/`LLMAuthenticationError` exceptions
- LLM narrative generation from ReportingAgent
- 45 LLM-specific tests

### Fixed
- 17 blind excepts (BLE001) — added noqa to intentional catch-alls
- 13 error message patterns (TRY003, EM101, EM102) — extracted messages to variables
- 89 auto-fixable ruff violations (D413, COM812, RUF100, FURB110, RUF022)
- 25 mypy type errors across 9 files
- 5 broken tests (feature engineering, target detection)
- Cascading .copy() calls in feature engineering pipeline
- Double os.path.getsize() in upload agent
- Qdrant upsert using range(len()) instead of zip()
