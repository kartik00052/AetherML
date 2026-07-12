"""AetherML REST API — thin HTTP layer over ``aetherml.run_pipeline()``.

No business logic lives here — request validation, error mapping, and
middleware are handled by FastAPI.  Run with::

    uvicorn aetherml.interfaces.api.app:app --reload
"""
