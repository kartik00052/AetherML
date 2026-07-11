<div align="center">

# AetherML

### An Open-Source Agentic Machine Learning Framework

**Transform raw datasets into production-ready machine learning workflows through intelligent multi-agent orchestration.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)
[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](#)
[![Downloads](https://img.shields.io/badge/downloads-0%2Fmonth-lightgrey.svg)](#)

<!-- ![AetherML Logo](docs/assets/logo.png) -->

</div>

---

## Table of Contents

1. [Why AetherML?](#why-aetherml)
2. [Key Features](#key-features)
3. [Architecture Overview](#architecture-overview)
4. [Complete Workflow](#complete-workflow)
5. [Folder Structure](#folder-structure)
6. [Multi-Agent Architecture](#multi-agent-architecture)
7. [Data Engine Abstraction](#data-engine-abstraction)
8. [SDK Usage](#sdk-usage)
9. [CLI Usage](#cli-usage)
10. [Future FastAPI Interface](#future-fastapi-interface)
11. [Plugin Architecture](#plugin-architecture)
12. [Technology Stack](#technology-stack)
13. [Installation](#installation)
14. [Development Guide](#development-guide)
15. [Testing Philosophy](#testing-philosophy)
16. [Design Principles](#design-principles)
17. [Roadmap](#roadmap)
18. [Contributing](#contributing)
19. [FAQ](#faq)
20. [License](#license)
21. [Acknowledgements](#acknowledgements)

---

## Why AetherML?

Most machine learning work today happens in one of two extremes.

At one end is the **notebook**: a flexible but fragile artifact where data validation, feature engineering, model selection, and reporting logic are interleaved with exploratory code. Notebooks are excellent for exploration and terrible for production. There is no enforced structure, no reusable state, and no clean boundary between "what the analyst did" and "what the system does."

At the other end is the **AutoML black box**: upload a dataset, wait, and receive a model with little insight into *why* particular transformations, encodings, or algorithms were chosen. AutoML tools optimize for a leaderboard metric, not for an engineer's understanding of the pipeline that produced it. When something goes wrong in production, there is often no inspectable, versionable pipeline to debug.

AetherML exists to occupy the space between these two extremes.

**AetherML is a Python SDK** that formalizes the machine learning lifecycle — validation, profiling, ETL, exploratory data analysis, feature engineering, target detection, model recommendation, training, evaluation, explainability, and reporting — as a graph of cooperating agents, each with a well-defined responsibility and a well-defined contract with the rest of the system.

**How it differs from notebooks:** every stage of the pipeline is a discrete, testable, reusable unit of code operating on a shared, typed `WorkflowState`, rather than an untracked sequence of cells.

**How it differs from AutoML tools:** every decision an agent makes — which imputation strategy, which encoding, which model family — is inspectable, overridable, and explainable. AetherML recommends; it does not obscure.

**How it differs from traditional hand-rolled pipelines:** the orchestration layer (built on LangGraph) manages state transitions, retries, and conditional branching between agents, so pipeline authors do not need to hand-write glue code for every new dataset or use case.

AetherML is **SDK-first**. The CLI, the (planned) FastAPI service, and any future desktop or web interface are thin clients built on top of the same SDK that a data scientist would `import` directly into their own scripts. There is exactly one source of truth for ML logic, and it is the SDK.

---

## Key Features

| Feature | Description | Status |
|---|---|---|
| **Multi-Agent Workflow** | Each pipeline stage (validation, EDA, feature engineering, etc.) is implemented as an independent agent with a single responsibility. | Implemented |
| **LangGraph Orchestration** | Agents are nodes in a directed graph; LangGraph manages state passing, conditional edges, and retries between them. | Implemented |
| **Automatic Engine Selection** | The framework inspects dataset size and shape to automatically choose between Pandas, Polars, and PySpark. | Implemented |
| **ETL** | Declarative extraction, cleaning, and transformation of raw tabular data into a canonical internal representation. | Implemented |
| **Validation** | Schema, type, and quality validation before any downstream processing occurs. | Implemented |
| **Exploratory Data Analysis (EDA)** | Automated statistical profiling and structured summaries of dataset characteristics. | Implemented |
| **Feature Engineering** | Automated and configurable transformation, encoding, and derivation of features. | Implemented |
| **Target Detection** | Heuristic and configurable identification of the likely prediction target and task type (classification/regression). | Implemented |
| **Model Recommendation** | Rule- and metric-driven suggestion of candidate model families based on data characteristics. | Implemented |
| **Explainability** | Post-training analysis producing feature importance and model-behavior summaries. | Implemented |
| **Reporting** | Structured, versionable output artifacts summarizing every stage of the run. | Implemented |
| **Modular Architecture** | Clear separation between agents, services, engines, and interfaces; no layer reaches across another's boundary. | Implemented |
| **Plugin System** | Extension points for custom agents, models, engines, and storage backends without modifying core code. | Planned |
| **Offline-First Design** | Core pipeline stages run without requiring network access or hosted services. | Implemented |
| **SDK-First Philosophy** | Every interface (CLI, API, GUI) is a client of the SDK — never a place where business logic lives. | Implemented |

---

## Architecture Overview

AetherML is organized into distinct layers. Each layer only depends on the layer(s) beneath it, never on layers above it or on sibling internals it doesn't own.

```
                     ┌────────────────────────┐
                     │        Python SDK        │   ← Public entry point (aetherml.SDK)
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │     LangGraph Workflow    │   ← Orchestrates agent execution order
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │          Agents           │   ← One responsibility per agent
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │         Services          │   ← Reusable domain logic, engine-agnostic
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │       Data Engines         │   ← Pandas / Polars / PySpark implementations
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   Reports / Storage        │   ← Persisted artifacts and run outputs
                     └────────────────────────┘
```

**Layer responsibilities:**

- **Python SDK** — The single public entry point. Exposes a high-level `SDK` (or `AetherML`) class that end users import and call. Hides all internal orchestration details.
- **LangGraph Workflow** — Defines the pipeline as a directed graph of nodes (agents) and edges (transitions, including conditional branches such as "skip EDA if `--fast` is set"). Owns the shared `WorkflowState` object that flows through the graph.
- **Agents** — Each agent performs exactly one pipeline responsibility (e.g., `ValidationAgent`, `EDAAgent`, `FeatureEngineeringAgent`). Agents read from and write to `WorkflowState`; they never talk to each other directly.
- **Services** — Stateless, reusable domain logic invoked by agents (e.g., statistical computations, encoding strategies, model scoring utilities). Services are engine-agnostic — they operate through the Data Engine abstraction rather than importing Pandas/Polars/PySpark directly.
- **Data Engines** — Concrete implementations of a common `DataEngine` interface for Pandas, Polars, and PySpark. Selected automatically or explicitly configured.
- **Reports / Storage** — Where run artifacts (validation reports, EDA summaries, trained model metadata, explainability output) are persisted, either to local disk (offline-first default) or to a configured backend.

---

## Complete Workflow

AetherML models the full ML lifecycle as a linear-with-branches pipeline. Each stage below is a node in the LangGraph workflow and corresponds to one or more agents.

```
 Dataset Upload
       │
       ▼
   Validation          → Schema/type/quality checks; halts pipeline on critical errors
       │
       ▼
   Profiling           → Lightweight structural summary (shape, dtypes, missingness)
       │
       ▼
     ETL                → Cleaning, normalization, canonicalization
       │
       ▼
     EDA                → Statistical analysis, distributions, correlations
       │
       ▼
Feature Engineering     → Encoding, scaling, derived features
       │
       ▼
Target Detection        → Identify prediction target and task type
       │
       ▼
Model Recommendation    → Candidate model families ranked by dataset fit
       │
       ▼
   Training              → Fit selected/recommended model(s)
       │
       ▼
   Evaluation            → Metric computation against held-out data
       │
       ▼
Explainability          → Feature importance, model behavior summary
       │
       ▼
   Reporting              → Persist structured run artifacts
```

**Stage summaries:**

- **Dataset Upload** — The entry point where a user provides a path, DataFrame, or supported data source reference.
- **Validation** — Confirms the dataset conforms to structural and quality expectations before any further processing is attempted; failures here stop the pipeline early rather than propagating bad data downstream.
- **Profiling** — A fast, lightweight pass that captures shape, dtypes, and missing-value structure, used to inform engine selection and later stages.
- **ETL** — Extract-transform-load logic that cleans, deduplicates, and normalizes the dataset into AetherML's canonical internal representation.
- **EDA** — Produces statistical summaries (distributions, correlations, outlier signals) consumed both by later agents and by the final report.
- **Feature Engineering** — Applies transformations, encodings, and derived-feature logic, configurable or automatic.
- **Target Detection** — Determines the most probable prediction target column and whether the task is classification or regression, using heuristics that are always overridable by explicit configuration.
- **Model Recommendation** — Scores candidate model families against the dataset's characteristics and produces a ranked shortlist.
- **Training** — Fits the recommended or user-selected model(s) on the engineered feature set.
- **Evaluation** — Computes task-appropriate metrics against a held-out split.
- **Explainability** — Generates feature importance and model-behavior summaries to support interpretation of results.
- **Reporting** — Serializes every stage's output into a structured, versioned artifact for later inspection or auditing.

---

## Folder Structure

The folder structure is the clearest expression of AetherML's architecture. Every directory below has a single, well-defined purpose. New contributors should be able to navigate this section instead of reading the entire source tree to understand where a given piece of logic belongs.

```
aetherml/
├── __init__.py          # Public SDK surface
├── exceptions.py        # Exception hierarchy
├── agents/              # Pipeline agents
│   ├── base.py          # BaseAgent, AgentResult, Tool
│   ├── upload/          # Data loading agent
│   ├── etl/             # ETL cleaning agent
│   ├── validation/      # Data validation agent
│   ├── eda/             # Exploratory data analysis
│   ├── target_detection/# Target column detection
│   ├── feature_engineering/ # Feature engineering
│   ├── model_selection/ # Model recommendation & training
│   ├── evaluation/      # Model evaluation
│   ├── explainability/  # SHAP-based explainability
│   ├── reporting/       # Report assembly
│   └── ...              # Additional agents
├── configs/             # Pydantic configuration
├── data/                # Data loading, validation, profiling
├── database/            # Qdrant vector store client
├── engines/             # Pandas/Polars/Spark engine abstraction
├── interfaces/          # CLI and future API
├── ml/                  # ML pipeline components
│   ├── automl/          # AutoML model selection
│   ├── evaluation/      # Metrics and evaluation
│   ├── explainability/  # SHAP explanations
│   ├── feature_engineering/ # Feature engineering
│   ├── reports/         # Report builder and templates
│   └── target_detection/ # Target detection heuristics
├── rag/                 # RAG infrastructure
└── workflow/            # LangGraph workflow orchestration
```

### `agents/`

**Purpose:** Houses every pipeline agent — the units of work that perform one pipeline responsibility each (validation, EDA, feature engineering, target detection, model recommendation, explainability, reporting, and so on).

**Responsibilities:** An agent reads relevant fields from the shared `WorkflowState`, invokes domain logic (from `ml/`, `data/`, or `engines/`) to do the actual computation, and writes its results back onto `WorkflowState`. Agents contain orchestration-adjacent logic (e.g., "should I skip this step given the current state?") but delegate all substantial domain logic to their respective modules.

**Key classes/modules:** `base.py` (defines the `BaseAgent` interface every agent implements), `validation_agent.py`, `profiling_agent.py`, `etl_agent.py`, `eda_agent.py`, `feature_engineering_agent.py`, `target_detection_agent.py`, `model_recommendation_agent.py`, `training_agent.py`, `evaluation_agent.py`, `explainability_agent.py`, `reporting_agent.py`.

**Why it exists:** Isolating each pipeline stage into its own agent keeps responsibilities small, testable in isolation, and independently replaceable — a contributor can rewrite the `FeatureEngineeringAgent` internals without touching validation or training logic.

**Interaction with other modules:** Agents are invoked exclusively by `workflow/` as graph nodes. They call into `services/` and `engines/` (indirectly, through services) and never import from `cli/`, `api/`, or `interfaces/`.

**Future extensions:** New agents (e.g., a `DriftDetectionAgent` or `FairnessAuditAgent`) can be added here and registered as new graph nodes without modifying existing agents.

### `workflow/`

**Purpose:** Defines the LangGraph graph itself — the nodes (agents), edges (transitions), conditional branching logic, and the shared `WorkflowState` schema.

**Responsibilities:** Wiring agents together in the correct order, handling conditional skips (e.g., "skip EDA if the user requested a fast run"), managing retries, and exposing a single `run()` entry point that the SDK calls.

**Key classes/modules:** `graph.py` (graph construction), `state.py` (`WorkflowState` definition, typically a typed dict or dataclass), `edges.py` (conditional transition logic).

**Why it exists:** Separating orchestration from agent logic means the pipeline's *shape* (what runs, in what order, under what conditions) can change without touching any individual agent's internals.

**Interaction with other modules:** Imports from `agents/`; is imported by the top-level SDK. Does not import from `engines/` or `services/` directly — all engine interaction happens inside agents/services.

**Future extensions:** Support for parallel branches (e.g., training multiple candidate models concurrently), human-in-the-loop checkpoints, and resumable/interruptible runs.

### `engines/`

**Purpose:** Contains the `DataEngine` abstraction and its concrete implementations for Pandas, Polars, and PySpark.

**Responsibilities:** Provide a uniform interface for reading data, performing transformations, and computing statistics, regardless of which underlying library is doing the work.

**Key classes/modules:** `base.py` (`DataEngine` abstract interface), `pandas_engine.py`, `polars_engine.py`, `pyspark_engine.py`, `factory.py` (engine selection logic based on dataset size/shape or explicit configuration).

**Why it exists:** Different dataset sizes call for different tools — Pandas for small-to-medium data, Polars for larger single-machine workloads, PySpark for distributed data. Centralizing this choice behind one interface means the rest of the codebase never has to know or care which engine is active.

**Interaction with other modules:** Used exclusively through `services/`, never imported directly by `agents/`, `cli/`, or `api/`. This boundary is intentional — see [Data Engine Abstraction](#data-engine-abstraction).

**Future extensions:** Additional engines (e.g., Dask, DuckDB) can be added by implementing `DataEngine` and registering with the factory.

### `services/`

**Purpose:** Stateless domain logic shared across agents — statistical computations, encoding strategies, scoring utilities, imputation logic, and similar reusable functions.

**Responsibilities:** Implement the actual "how" behind each agent's "what," using the `DataEngine` abstraction rather than any specific dataframe library.

**Key classes/modules:** `validation_service.py`, `profiling_service.py`, `eda_service.py`, `feature_engineering_service.py`, `model_scoring_service.py`, `explainability_service.py`.

**Why it exists:** Keeps agents thin and orchestration-focused while housing substantial, independently testable domain logic in one place, reusable across multiple agents if needed.

**Interaction with other modules:** Called by `agents/`; internally calls `engines/` through the `DataEngine` interface.

**Future extensions:** New statistical methods, encoding strategies, or scoring functions can be added here without touching agent code.

### `ml/`

**Purpose:** Houses model definitions, training routines, and evaluation metric implementations.

**Responsibilities:** Wraps underlying ML libraries (e.g., scikit-learn) behind AetherML's own model interface so agents interact with a consistent API regardless of the underlying model implementation.

**Key classes/modules:** `models/` (model family wrappers), `metrics.py` (evaluation metric implementations), `trainer.py` (training loop orchestration invoked by `TrainingAgent`).

**Why it exists:** Decouples the framework's model-selection and training logic from any single ML library, making it possible to add new model backends without changing agent code.

**Interaction with other modules:** Invoked by `ModelRecommendationAgent`, `TrainingAgent`, and `EvaluationAgent`.

**Future extensions:** Support for additional model families, hyperparameter search strategies, and custom model registration via the plugin system.

### `configs/`

**Purpose:** Centralizes configuration schemas and default values for the entire pipeline (engine selection thresholds, validation rules, feature engineering defaults, model recommendation weights, etc.).

**Responsibilities:** Define and validate configuration objects passed into the SDK, ensuring every stage behaves predictably and every override is explicit and typed.

**Key classes/modules:** `pipeline_config.py`, `engine_config.py`, `defaults.py`.

**Why it exists:** Keeps "magic numbers" and default behavior out of business logic, making the framework's default behavior auditable and its overrides discoverable in one place.

**Interaction with other modules:** Read by `workflow/`, `agents/`, and `services/` at run time; never mutated during a run.

**Future extensions:** Per-environment config profiles (dev/staging/prod), YAML/JSON config file loading.

### `exceptions/`

**Purpose:** Defines AetherML's exception hierarchy.

**Responsibilities:** Provide specific, catchable exception types (e.g., `ValidationError`, `EngineSelectionError`, `TargetDetectionError`) rather than relying on generic Python exceptions, so callers of the SDK can handle failure modes precisely.

**Key classes/modules:** `base.py` (`AetherMLError` root exception), `validation_exceptions.py`, `engine_exceptions.py`, `workflow_exceptions.py`.

**Why it exists:** Precise exception types make SDK error handling predictable for downstream users and make internal failure paths self-documenting.

**Interaction with other modules:** Raised throughout `agents/`, `services/`, and `engines/`; caught and handled at the SDK boundary and by `cli/`.

**Future extensions:** Structured error codes for programmatic handling by the future FastAPI interface.

### `interfaces/`

**Purpose:** Defines the abstract base classes and protocols that concrete implementations (agents, engines, storage backends) must satisfy.

**Responsibilities:** Establish contracts (e.g., `BaseAgent`, `DataEngine`, `StorageBackend`) independent of any specific implementation, enabling the plugin system and dependency injection throughout the framework.

**Key classes/modules:** `agent_interface.py`, `engine_interface.py`, `storage_interface.py`, `report_interface.py`.

**Why it exists:** Encodes AetherML's commitment to programming against interfaces rather than concrete implementations, which is what makes the plugin architecture and engine abstraction possible.

**Interaction with other modules:** Implemented by classes throughout `agents/`, `engines/`, and `storage/`; referenced (not implemented) by `workflow/`.

**Future extensions:** Additional interfaces as new extension points (custom report formats, custom storage backends) are introduced.

### `cli/`

**Purpose:** The command-line interface built on top of the SDK.

**Responsibilities:** Parse command-line arguments, translate them into SDK calls, and format SDK output for terminal display. Contains no business logic of its own.

**Key classes/modules:** `main.py` (entry point), `commands/` (subcommand implementations, e.g., `run.py`, `validate.py`).

**Why it exists:** Provides a convenient, scriptable way to invoke the SDK without writing Python, while keeping all real logic in the SDK itself.

**Interaction with other modules:** Imports only from the top-level SDK; never reaches into `agents/`, `services/`, or `engines/` directly.

**Future extensions:** Interactive mode, shell completion, richer terminal output.

### `api/`

**Purpose:** Reserved for the **planned** FastAPI interface that will expose the SDK over HTTP.

**Responsibilities (planned):** Translate HTTP requests into SDK calls and SDK results into HTTP responses. Like the CLI, it is intended to contain no business logic.

**Key classes/modules (planned):** `main.py`, `routers/`, `schemas/` (Pydantic request/response models).

**Why it exists:** To provide programmatic, network-accessible access to the SDK for teams building services or UIs on top of AetherML.

**Status:** **Planned** — not yet implemented. See [Future FastAPI Interface](#future-fastapi-interface).

### `reports/`

**Purpose:** Defines the structure and generation logic for AetherML's output reports (validation reports, EDA summaries, model evaluation reports, explainability reports).

**Responsibilities:** Aggregate `WorkflowState` data at the end of a run (or at intermediate checkpoints) into structured, human- and machine-readable report artifacts.

**Key classes/modules:** `report_builder.py`, `templates/` (report formatting logic), `schema.py` (report data schema).

**Why it exists:** Every AetherML run should produce an auditable, inspectable record of what happened and why, independent of whether the user is watching the terminal in real time.

**Interaction with other modules:** Invoked by `ReportingAgent`; writes through `storage/`.

**Future extensions:** HTML/PDF report rendering, report diffing between runs.

### `storage/`

**Purpose:** Abstracts *where* artifacts (reports, trained model metadata, intermediate datasets) are persisted.

**Responsibilities:** Provide a uniform storage interface with a local-filesystem implementation as the offline-first default, with room for additional backends.

**Key classes/modules:** `base.py` (`StorageBackend` interface), `local_storage.py`.

**Why it exists:** Keeps the rest of the framework agnostic to storage location, supporting AetherML's offline-first design while leaving room for cloud storage backends later.

**Interaction with other modules:** Used by `reports/` and `ml/` (for persisting trained model artifacts).

**Future extensions:** S3/GCS/Azure Blob backends, database-backed run history.

### `plugins/`

**Purpose:** Reserved for the **planned** plugin system that will let third parties extend agents, models, data engines, reports, and storage backends without modifying core framework code.

**Responsibilities (planned):** Discover, load, and validate externally registered plugins against the `interfaces/` contracts.

**Status:** **Planned** — not yet implemented. See [Plugin Architecture](#plugin-architecture).

### `tests/`

**Purpose:** Houses the full test suite.

**Responsibilities:** Verify correctness of individual units (`unit/`), interaction between components (`integration/`), stability of pipeline outputs over time (`regression/`), and adherence to architectural boundaries (`architecture/`).

**Key classes/modules:** `unit/`, `integration/`, `regression/`, `architecture/`, `conftest.py` (shared fixtures).

**Why it exists:** See [Testing Philosophy](#testing-philosophy) for the full rationale behind each test category.

**Interaction with other modules:** Imports from every other package as needed; is never imported by production code.

### `docs/`

**Purpose:** Source for the project's documentation site (architecture guides, API reference, tutorials).

**Responsibilities:** House long-form documentation that doesn't belong in the README, including detailed agent-by-agent references and configuration guides.

**Key classes/modules:** `architecture/`, `guides/`, `api-reference/`, `assets/` (images, diagrams referenced by this README and the docs site).

**Why it exists:** Keeps the README focused on onboarding and overview while deeper reference material lives in a dedicated, searchable location.

### `examples/`

**Purpose:** Runnable, minimal example scripts demonstrating SDK usage.

**Responsibilities:** Provide copy-pasteable, working examples for common use cases (full pipeline run, running a subset of stages, custom configuration, custom agents).

**Key classes/modules:** `basic_analysis.py`, `custom_config.py`, `selected_stages.py`.

**Why it exists:** Examples are often the first code a new user reads; keeping them runnable and current is treated as a first-class documentation responsibility.

---

## Multi-Agent Architecture

Every agent implements the same `BaseAgent` interface and interacts with the pipeline exclusively through the shared `WorkflowState`. No agent calls another agent directly — all sequencing is owned by the `workflow/` graph.

| Agent | Responsibility | Inputs (from `WorkflowState`) | Outputs (to `WorkflowState`) | Downstream Consumers |
|---|---|---|---|---|
| **ValidationAgent** | Schema, type, and quality checks on the raw dataset | Raw dataset reference | Validation report, pass/fail status | `ProfilingAgent`, `ReportingAgent` |
| **ProfilingAgent** | Lightweight structural profiling (shape, dtypes, missingness) | Validated dataset | Profile summary | `EngineFactory` (via services), `ETLAgent` |
| **ETLAgent** | Cleaning, normalization, canonicalization | Validated dataset, profile summary | Canonical dataset | `EDAAgent`, `FeatureEngineeringAgent` |
| **EDAAgent** | Statistical analysis: distributions, correlations, outliers | Canonical dataset | EDA summary | `FeatureEngineeringAgent`, `ReportingAgent` |
| **FeatureEngineeringAgent** | Encoding, scaling, derived feature creation | Canonical dataset, EDA summary | Engineered feature set | `TargetDetectionAgent` |
| **TargetDetectionAgent** | Identify likely target column and task type | Engineered feature set | Target column, task type | `ModelRecommendationAgent` |
| **ModelRecommendationAgent** | Rank candidate model families | Engineered feature set, task type | Ranked model shortlist | `TrainingAgent` |
| **TrainingAgent** | Fit selected/recommended model(s) | Feature set, target, model choice | Trained model artifact | `EvaluationAgent` |
| **EvaluationAgent** | Compute task-appropriate metrics | Trained model, held-out split | Evaluation metrics | `ExplainabilityAgent`, `ReportingAgent` |
| **ExplainabilityAgent** | Feature importance and behavior summary | Trained model, feature set | Explainability summary | `ReportingAgent` |
| **ReportingAgent** | Aggregate all prior outputs into a persisted report | All prior `WorkflowState` fields | Final report artifact | `storage/` |

This table is intentionally architecture-focused: it describes *contracts*, not implementation details. Full implementation-level documentation for each agent lives in `docs/architecture/agents/`.

---

## Data Engine Abstraction

AetherML supports three dataframe backends — **Pandas**, **Polars**, and **PySpark** — because no single library is optimal across the full range of dataset sizes AetherML is meant to handle: Pandas is ideal for small-to-medium in-memory datasets, Polars offers substantially better performance on larger single-machine workloads, and PySpark is necessary once data exceeds what a single machine can comfortably hold.

**Selection strategy:** `ProfilingAgent` inspects the dataset's row count, column count, and estimated memory footprint, and passes this information to `engines/factory.py`, which selects the appropriate engine according to configurable thresholds in `configs/engine_config.py`. Users may also force a specific engine explicitly through SDK configuration, overriding automatic selection.

**Engine interface:** Every engine implements the same `DataEngine` abstract interface defined in `engines/base.py` — covering operations such as reading data, filtering, aggregating, computing summary statistics, and writing output. Agents and services never call Pandas, Polars, or PySpark APIs directly; they call `DataEngine` methods, and the concrete engine implementation handles the translation.

**Factory pattern:** `engines/factory.py` is the single place in the codebase responsible for instantiating a concrete `DataEngine`. This is a textbook Factory pattern: callers ask for "a data engine appropriate for this dataset" and receive a concrete implementation without needing to know which one they got.

**Why other modules never import engine-specific libraries directly:** This boundary is what allows AetherML to add a new engine (say, DuckDB) in the future by implementing one new class, with zero changes required to `agents/`, `services/`, `ml/`, `cli/`, or the SDK's public API. It also makes it possible to unit test agent and service logic against a lightweight in-memory fake `DataEngine`, without requiring Polars or PySpark to be installed for most of the test suite.

---

## SDK Usage

AetherML is designed to be used as an imported library first, and a CLI second. All of the examples below assume `pip install aetherml` (or an editable install — see [Installation](#installation)).

### Basic Analysis

```python
import asyncio
from aetherml import run_pipeline

async def main():
    result = await run_pipeline(data_path="data/customers.csv")
    print(result)

asyncio.run(main())
```

### Running Selected Stages

```python
import asyncio
from aetherml import run_pipeline

async def main():
    result = await run_pipeline(
        data_path="data/customers.csv",
        stages=["upload", "etl", "validation"],
    )
    print(result)

asyncio.run(main())
```

### Custom Configuration

```python
import asyncio
from aetherml import AetherMLConfig, run_pipeline

async def main():
    config = AetherMLConfig()
    config.engine.preferred = "polars"  # force Polars instead of auto-selection

    result = await run_pipeline(
        data_path="data/customers.csv",
        config=config,
    )
    print(result)

asyncio.run(main())
```

### Error Handling

```python
import asyncio
from aetherml import run_pipeline
from aetherml.exceptions import DataValidationError, EngineSelectionError, WorkflowError

async def main():
    try:
        result = await run_pipeline(data_path="data/customers.csv")
    except DataValidationError as e:
        print(f"Dataset failed validation: {e}")
    except EngineSelectionError as e:
        print(f"Could not select a data engine: {e}")
    except WorkflowError as e:
        print(f"Pipeline failed: {e}")

asyncio.run(main())
```

> **Note:** Custom agent injection depends on the plugin system, which is currently **planned**, not implemented. This example illustrates the intended future API shape.

---

## CLI Usage

The CLI is a thin wrapper around the SDK, intended for quick, scriptable runs without writing Python.

```bash
# Run the full pipeline on a dataset
aetherml run data/customers.csv

# Force a specific data engine
aetherml run data/customers.csv --engine polars

# View SDK info
aetherml info
```

Each command maps directly onto an SDK call — `aetherml run` calls `AetherML.run(...)`, `aetherml validate` calls the validation stage in isolation. No command contains logic that isn't already present in the SDK.

---

## Future FastAPI Interface

> **Status: Planned.** The `api/` directory is currently a placeholder; no FastAPI service exists yet.

The intended design, once implemented, mirrors the CLI's relationship to the SDK: **FastAPI will contain no business logic.** Every route handler will do nothing more than parse the incoming request into SDK call parameters, invoke the SDK, and serialize the SDK's result into an HTTP response.

Planned shape:

```python
# api/routers/run.py  (planned, not yet implemented)

from fastapi import APIRouter
from aetherml import AetherML
from api.schemas import RunRequest, RunResponse

router = APIRouter()
sdk = AetherML()

@router.post("/run", response_model=RunResponse)
def run_pipeline(request: RunRequest) -> RunResponse:
    result = sdk.run(
        dataset_path=request.dataset_path,
        stages=request.stages,
        config=request.config,
    )
    return RunResponse.from_sdk_result(result)
```

The rationale for this boundary: if business logic ever leaked into `api/`, the SDK would stop being the single source of truth, and behavior could silently diverge between users who call the SDK directly and users who call it over HTTP. Keeping `api/` deliberately "dumb" is a design constraint, not an oversight.

---

## Plugin Architecture

> **Status: Planned.** The `plugins/` directory and the plugin-loading mechanism described below do not yet exist in the codebase.

AetherML's `interfaces/` module already defines the contracts (`BaseAgent`, `DataEngine`, `StorageBackend`, and future `ModelBackend`/`ReportFormat` interfaces) that make a plugin system possible. The planned plugin architecture will allow third parties to extend the framework in the following ways, without modifying core framework code:

- **Agents** — Register a custom agent (implementing `BaseAgent`) as an additional or replacement node in the workflow graph.
- **Models** — Register a custom model backend (implementing a future `ModelBackend` interface) to be considered during model recommendation and training.
- **Data Engines** — Register a new `DataEngine` implementation (e.g., DuckDB) alongside Pandas, Polars, and PySpark.
- **Reports** — Register a custom report format or renderer implementing a future `ReportFormat` interface.
- **Storage** — Register a custom `StorageBackend` (e.g., S3, GCS) implementing the existing `StorageBackend` interface.

The mechanism under consideration is a Python entry-points-based discovery system, similar in spirit to how pytest and Flake8 discover plugins — packages installed alongside AetherML that declare an `aetherml.plugins` entry point will be automatically discovered and validated against the relevant `interfaces/` contract at startup.

---

## Technology Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI%20(planned)-009688?style=for-the-badge&logo=fastapi&logoColor=white)

</div>

| Technology | Role in AetherML | Why It Was Chosen |
|---|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core implementation language | Mature ML/data ecosystem; type hints support the interface-driven design |
| ![LangGraph](https://img.shields.io/badge/-LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white) | Workflow orchestration engine | Purpose-built for modeling stateful, graph-shaped agent workflows with conditional branching |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Default data engine for small-to-medium datasets | Ubiquitous, well-understood, ideal for datasets that fit comfortably in memory |
| ![Polars](https://img.shields.io/badge/-Polars-CD792C?style=flat-square&logo=polars&logoColor=white) | Data engine for larger single-machine workloads | Significantly faster than Pandas on larger datasets due to its Rust-based, multi-threaded query engine |
| ![PySpark](https://img.shields.io/badge/-PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white) | Data engine for distributed/large-scale datasets | Industry-standard for data that exceeds single-machine memory |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Underlying model implementations in `ml/` | Battle-tested, consistent API surface for the model families AetherML wraps |
| ![Pydantic](https://img.shields.io/badge/-Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white) | Configuration and (planned) API schema validation | Enforces typed, validated configuration objects rather than untyped dicts |
| ![pytest](https://img.shields.io/badge/-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white) | Test runner across all test categories | De facto standard for Python testing; strong fixture and plugin ecosystem |
| ![FastAPI](https://img.shields.io/badge/-FastAPI%20(planned)-009688?style=flat-square&logo=fastapi&logoColor=white) | HTTP interface layer | Async-first, automatic OpenAPI schema generation, thin enough to stay logic-free |

---

## Installation

**Requirements:** Python 3.11 or newer.

### Standard Installation

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install aetherml
```

### Development Installation (Editable)

For contributors who want to modify AetherML itself:

```bash
git clone https://github.com/your-org/aetherml.git
cd aetherml

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

The `[dev]` extra installs testing, linting, and formatting tools in addition to AetherML's runtime dependencies. The editable (`-e`) install links the installed package directly to your working copy of the source, so changes take effect immediately without reinstalling.

---

## Development Guide

### Repository Setup

```bash
git clone https://github.com/your-org/aetherml.git
cd aetherml
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest                        # full suite
pytest tests/unit             # unit tests only
pytest tests/integration       # integration tests only
pytest -k "validation"        # tests matching a keyword
```

### Linting

```bash
ruff check .
```

### Formatting

```bash
ruff format .
```

### Contribution Workflow

1. Fork the repository and create a feature branch off `main`.
2. Make your changes, ensuring new code is covered by tests.
3. Run the full test suite and linter locally before opening a pull request.
4. Open a pull request describing the change, referencing any related issue.
5. Address review feedback; a maintainer will merge once approved.

---

## Testing Philosophy

AetherML's test suite is organized into four categories, each answering a different question:

| Category | Question It Answers | Example |
|---|---|---|
| **Unit Tests** | Does this single function/class behave correctly in isolation? | Does `EncodingService.one_hot()` produce the expected columns? |
| **Integration Tests** | Do multiple components work correctly together? | Does `ValidationAgent → ProfilingAgent → ETLAgent` produce a correctly canonicalized dataset? |
| **Regression Tests** | Does the pipeline continue to produce stable, expected output over time? | Does a fixed sample dataset always produce the same EDA summary across releases? |
| **Architecture Tests** | Are the module boundaries described in this README actually enforced? | Does static analysis confirm that `agents/` never imports from `engines/` directly? |

Architecture tests are a deliberate choice: they encode the layering rules described in [Architecture Overview](#architecture-overview) and [Data Engine Abstraction](#data-engine-abstraction) as executable checks, so that boundary violations are caught in CI rather than discovered later during a refactor.

---

## Design Principles

AetherML's internal structure is guided by a consistent set of principles:

- **SDK-first** — The SDK is the single source of truth for all ML logic. CLI, API, and any future GUI are clients, not implementers, of behavior.
- **Offline-first** — Core pipeline stages run without requiring network access; storage defaults to the local filesystem.
- **Deterministic ML** — Given the same input data and configuration, a pipeline run should be reproducible; randomness (e.g., train/test splitting, model initialization) is seeded explicitly.
- **Dependency Injection** — Agents and services receive their dependencies (data engines, storage backends) rather than constructing them internally, which is what makes testing with fakes/mocks practical.
- **Clean Architecture** — Layers depend inward (interfaces → implementations), never outward; see [Architecture Overview](#architecture-overview).
- **SOLID** — In particular, Single Responsibility (one agent, one job) and Interface Segregation (`interfaces/` defines narrow, focused contracts).
- **Strategy Pattern** — Used for interchangeable behaviors such as data engine selection and (planned) model backend selection.
- **Factory Pattern** — Used for engine instantiation (`engines/factory.py`) and, in the future, plugin instantiation.
- **Composition over Inheritance** — Agents compose services rather than inheriting shared behavior through deep class hierarchies.
- **Plugin-friendly design** — Even before the plugin system itself is implemented, every extension point is designed against an interface in `interfaces/`, so plugins can be added later without retrofitting the core.

---

## Roadmap

![Status: Active Development](https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square)
![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square)

### Completed

- [x] Core `WorkflowState` and LangGraph-based orchestration
- [x] Validation, Profiling, ETL, EDA, Feature Engineering, Target Detection, Model Recommendation, Training, Evaluation, Explainability, and Reporting agents
- [x] Pandas, Polars, and PySpark data engines with automatic selection
- [x] Local filesystem storage backend
- [x] CLI interface
- [x] Unit, integration, regression, and architecture test suites

### Planned

- [ ] FastAPI HTTP interface (`api/`)
- [ ] Plugin system with entry-points-based discovery (`plugins/`)
- [ ] Additional storage backends (S3, GCS, Azure Blob)
- [ ] Additional data engine support (DuckDB)
- [ ] HTML/PDF report rendering
- [ ] Parallel/branching agent execution within the workflow graph
- [ ] Desktop GUI client built on top of the SDK
- [ ] Human-in-the-loop checkpoints within the workflow graph

---

## Contributing

![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)
![Good First Issues](https://img.shields.io/github/issues/your-org/aetherml/good%20first%20issue?style=flat-square&label=good%20first%20issues)

Contributions of all kinds are welcome — bug fixes, new agents, documentation improvements, and test coverage.

**Coding standards:** Code must pass `ruff check .` and `ruff format .` before review. New public functions and classes should include type hints and docstrings. New behavior should be accompanied by tests in the appropriate category (see [Testing Philosophy](#testing-philosophy)).

**Branch naming:** Use `feature/<short-description>`, `fix/<short-description>`, or `docs/<short-description>` as appropriate.

**Pull Requests:** Keep PRs focused on a single change where possible. Reference any related issue in the PR description. Ensure CI passes before requesting review.

**Issues:** Use issues to report bugs or propose new features. Please include a minimal reproducible example for bug reports.

**Discussions:** For open-ended design questions (e.g., "how should the plugin system's discovery mechanism work?"), use GitHub Discussions rather than an issue, to keep design conversations separate from actionable bug/feature tracking.

---

## FAQ

**Why not just use AutoML?**
AutoML tools optimize for a leaderboard metric and often hide the reasoning behind their choices. AetherML is designed so every decision — imputation strategy, encoding, model family — is inspectable and overridable, at the cost of being less "hands-off" than a pure AutoML tool.

**Why LangGraph specifically?**
LangGraph models workflows as a graph of stateful nodes with conditional edges, which maps directly onto AetherML's pipeline shape (sequential stages with occasional conditional skips), without requiring hand-written orchestration and retry logic.

**Why Polars in addition to Pandas?**
Pandas is the default for small-to-medium datasets, but its single-threaded execution model becomes a bottleneck on larger data. Polars' multi-threaded, Rust-based engine handles larger single-machine workloads significantly faster, so AetherML automatically upgrades to Polars when dataset size warrants it.

**Why SDK-first instead of API-first or app-first?**
Building the SDK first ensures there is exactly one place where ML logic lives. Every interface built afterward (CLI, and the planned FastAPI service and GUI) is a client of that logic rather than a second implementation of it, which avoids behavioral drift between interfaces.

**Can I use only the ETL stage without running the full pipeline?**
Yes. The SDK's `run()` method accepts a `stages` parameter that lets you run any subset of the pipeline — see [SDK Usage](#sdk-usage).

**Can I integrate AetherML with FastAPI myself, today?**
Yes, informally — since the SDK is a plain Python package, you can already wrap `AetherML.run()` in your own FastAPI routes. The `api/` module described in this README is AetherML's own **planned**, first-party FastAPI interface, not a prerequisite for using AetherML from a FastAPI app you build yourself.

**Can I build my own agents?**
Not yet through a formal plugin mechanism — that is **planned** (see [Plugin Architecture](#plugin-architecture)). Today, extending the pipeline with a custom agent requires modifying `workflow/graph.py` directly in a fork or contribution.

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 AetherML Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See the [LICENSE](LICENSE) file for the full text.

---

## Acknowledgements

<div align="center">

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat-square&logo=polars&logoColor=white)

</div>

AetherML's design draws inspiration from the architectural patterns and developer experience established by several mature open-source projects, without any affiliation with or endorsement from them:

- **[scikit-learn](https://scikit-learn.org/)** — for its consistent estimator API and commitment to interpretable, well-documented behavior.
- **[FastAPI](https://fastapi.tiangolo.com/)** — for demonstrating how a thin, type-driven interface layer can sit cleanly on top of independent business logic.
- **[LangGraph](https://www.langchain.com/langgraph)** — for the graph-based orchestration model that AetherML's workflow layer is built directly on top of.
- **[MLflow](https://mlflow.org/)** — for its approach to structured, versionable run tracking and reporting.
- **[Polars](https://pola.rs/)** — for showing what a modern, high-performance dataframe engine can look like, and for being one of AetherML's actual data engines.

---

<div align="center">

Built with a commitment to transparent, inspectable machine learning pipelines.

</div>
