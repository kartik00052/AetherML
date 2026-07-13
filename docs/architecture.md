# Architecture

PhronesisML is built as a **graph of cooperating agents** — each stage of the ML pipeline is a discrete, testable unit operating on a shared typed state. This page explains how the system is structured and why.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SDK Layer                         │
│  Phronesis (OOP) │ Simple API │ run_pipeline()      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Workflow Layer (LangGraph)             │
│  StateGraph(WorkflowState) → compiled graph         │
│  Nodes: agent closures │ Router: conditional edges  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Agent Layer                          │
│  Upload │ ETL │ Validation │ EDA │ Target │ FE │ …  │
│  Each agent: Protocol-based, stateless, returns     │
│  AgentResult(success, data, error)                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Engine Layer (ABC)                      │
│  PandasEngine │ PolarsEngine │ SparkEngine           │
│  Uniform interface: read, collect, filter, join …   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Data Layer                           │
│  File loaders │ Validators │ Transformers │ Profiles │
└─────────────────────────────────────────────────────┘
```

---

## The Agent Pattern

### Protocol-Based Design

Agents do **not** inherit from a base class. They satisfy the `BaseAgent` protocol via duck typing (structural subtyping). This avoids diamond-inheritance issues and keeps agents independently testable.

```python
from typing import Protocol, Any

class BaseAgent(Protocol):
    """Agents must satisfy this protocol via duck typing."""

    def run(self, state: dict[str, Any]) -> AgentResult: ...
```

### AgentResult — Standardized Returns

Every agent returns an `AgentResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the agent completed successfully |
| `data` | `dict` | Stage-specific output (merged into WorkflowState) |
| `error` | `Exception \| None` | The exception if `success=False` |
| `error_type` | `str \| None` | Exception class name |
| `error_message` | `str \| None` | Human-readable error message |
| `error_context` | `dict \| None` | Structured context for debugging |

**Key rule:** Agents MUST NOT raise exceptions for expected failures. They return `AgentResult(success=False, ...)` and the workflow handles it gracefully.

### Stateless Agents

All mutable state lives in `WorkflowState`. Agents are pure functions — given the same state, they produce the same result. This makes them independently testable.

---

## LangGraph Integration

PhronesisML uses [LangGraph](https://langchain-ai.github.io/langgraph/) to model the pipeline as a stateful graph.

### StateGraph with Pydantic State

```python
from langgraph.graph import StateGraph
from phronesisml.workflow.state import WorkflowState

graph = StateGraph(WorkflowState)  # Pydantic BaseModel, not TypedDict
```

Using Pydantic for state provides runtime validation and serialization — every field is typed and validated at each stage.

### Linear Pipeline Topology

The canonical pipeline order:

```
upload → etl → validation → eda → target_detection →
feature_engineering → model_selection → evaluation →
explainability → reporting → storage
```

### Conditional Routing

Each stage has a routing function that returns `"proceed"` or `"__end__"`. If the stage produced its expected output, the pipeline continues. If not, it ends gracefully.

```python
def route_after_validation(state: WorkflowState) -> str:
    """Proceed if validation passed, end if data is invalid."""
    if state.validated_data is not None:
        return "proceed"
    return "__end__"
```

This "fail-safe" pattern means partial results are always available — if stage 5 fails, you still get results from stages 1–4.

### Graph Caching

Compiled graphs are cached by `(agent_names, stages, agent_ids)`. Repeated calls with the same topology skip LangGraph compilation — providing a 20–40% speedup on subsequent runs.

---

## Engine Abstraction

### BaseEngine ABC

All engines inherit from `BaseEngine(ABC)` with `@abstractmethod` for every core operation. This forces subclassing and ensures every new engine is audited against the full interface.

```python
class BaseEngine(ABC):
    @abstractmethod
    def read(self, path, **kwargs) -> Any: ...

    @abstractmethod
    def collect(self, df) -> pd.DataFrame: ...

    @abstractmethod
    def filter(self, df, conditions) -> Any: ...

    # ... 10+ more abstract methods
```

### Three-Tier Engine Selection

The engine is auto-selected based on data size:

| Data Size | Engine | Rationale |
|-----------|--------|-----------|
| < 2 MB | **Pandas** | Fast startup, familiar API |
| 2–500 MB | **Polars** | 2-10x faster, lower memory |
| > 500 MB | **Spark** | Distributed computing |

Users can override with `config.engine.preferred = "polars"`.

### Universal Pandas Output

`engine.collect()` always returns a `pd.DataFrame` regardless of backend. This means downstream code (profiling, evaluation, reporting) works uniformly — it doesn't need to know which engine is active.

---

## Composition Root

Agent instantiation happens in exactly two places:

1. **`_compose_agents()`** in `__init__.py` — for the Simple API
2. **`_make_agents()`** in `sdk.py` — for the OOP API

All other code depends on abstractions. This "composition root" pattern makes it easy to swap implementations or add dependency injection.

---

## Data Flow

```
User provides file path
        │
        ▼
    Upload Agent ──────► raw_data (DataFrame)
        │
        ▼
    ETL Agent ──────────► processed_data (DataFrame)
        │
        ▼
    Validation Agent ───► validated_data (DataFrame)
        │
        ▼
    EDA Agent ──────────► data_profile (dict)
        │
        ▼
    Target Detection ───► target_column, task_type, confidence
        │
        ▼
    Feature Engineering ► features (DataFrame), feature_names
        │
        ▼
    Model Selection ────► trained_model, best_pipeline
        │
        ▼
    Evaluation ─────────► evaluation_report (metrics dict)
        │
        ▼
    Explainability ─────► explanation (SHAP values)
        │
        ▼
    Reporting ──────────► final_report (Markdown string)
        │
        ▼
    Storage ────────────► artifacts saved to disk
```

Each arrow represents a state transition. The `WorkflowState` accumulates results — by the end, it contains everything from the raw data to the final report.

---

## Error Handling Hierarchy

```
PhronesisError (base)
├── ConfigurationError
├── DataError
│   ├── DataLoadError
│   ├── DataTransformError
│   └── DataValidationError
├── EngineError
│   └── EngineSelectionError
├── WorkflowError
└── AgentError
    └── AgentNotImplementedError
```

The system is designed for **graceful degradation**:

- If SHAP is not installed, explainability returns empty results — the pipeline continues.
- If an agent raises `AgentNotImplementedError`, the node logs a warning and returns `{}`.
- If MLflow is unreachable, training continues without logging.
- Partial results are always available even if later stages fail.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `phronesisml/sdk.py` | OOP API — `Phronesis` class |
| `phronesisml/simple.py` | Simple API — one-liner functions |
| `phronesisml/__init__.py` | Public exports, composition root |
| `phronesisml/configs/settings.py` | `PhronesisConfig` and sub-configs |
| `phronesisml/agents/base.py` | `BaseAgent` protocol, `AgentResult` |
| `phronesisml/workflow/graph.py` | LangGraph `StateGraph` construction |
| `phronesisml/workflow/state.py` | `WorkflowState` Pydantic model |
| `phronesisml/workflow/router.py` | Conditional routing functions |
| `phronesisml/engines/base_engine.py` | `BaseEngine` ABC |
| `phronesisml/engines/engine_selector.py` | Auto engine selection |
| `phronesisml/exceptions.py` | Exception hierarchy |
