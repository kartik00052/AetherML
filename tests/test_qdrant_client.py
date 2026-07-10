"""Tests for the Qdrant vector store client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherml.database.qdrant.client import QdrantClient


class TestQdrantClientInit:
    """Test QdrantClient initialisation."""

    def test_default_config(self) -> None:
        client = QdrantClient()
        assert client._url == "http://localhost:6333"
        assert client._api_key is None
        assert client._collection_name == "aetherml_knowledge"
        assert client._timeout_seconds == 5.0

    def test_custom_config(self) -> None:
        client = QdrantClient(
            url="http://custom:6334",
            api_key="secret",
            collection_name="my_collection",
            timeout_seconds=10.0,
        )
        assert client._url == "http://custom:6334"
        assert client._api_key == "secret"
        assert client._collection_name == "my_collection"
        assert client._timeout_seconds == 10.0

    def test_env_fallback_url(self) -> None:
        with patch.dict("os.environ", {"AETHERML_QDRANT_URL": "http://env:6333"}):
            client = QdrantClient(url="")
            assert client._url == "http://env:6333"

    def test_env_fallback_api_key(self) -> None:
        with patch.dict("os.environ", {"AETHERML_QDRANT_API_KEY": "env-key"}):
            client = QdrantClient()
            assert client._api_key == "env-key"


class TestQdrantClientImportError:
    """Test graceful degradation when qdrant-client is not installed."""

    def test_get_client_import_error(self) -> None:
        client = QdrantClient()
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(ImportError, match="qdrant-client is not installed"),
        ):
            client._get_client()


class TestQdrantClientEnsureCollection:
    """Test ensure_collection method."""

    def test_success_creates_collection(self) -> None:
        client = QdrantClient()
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_models = MagicMock()

        with (
            patch.object(client, "_get_client", return_value=mock_qdrant_instance),
            patch.dict("sys.modules", {"qdrant_client.models": mock_models}),
        ):
            result = client.ensure_collection(dimension=384)

        assert result is True
        mock_qdrant_instance.create_collection.assert_called_once()

    def test_success_already_exists(self) -> None:
        client = QdrantClient()
        mock_qdrant_instance = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "aetherml_knowledge"
        mock_qdrant_instance.get_collections.return_value.collections = [mock_collection]
        mock_models = MagicMock()

        with (
            patch.object(client, "_get_client", return_value=mock_qdrant_instance),
            patch.dict("sys.modules", {"qdrant_client.models": mock_models}),
        ):
            result = client.ensure_collection()

        assert result is True
        mock_qdrant_instance.create_collection.assert_not_called()

    def test_failure_returns_false(self) -> None:
        client = QdrantClient()
        with patch.object(client, "_get_client", side_effect=Exception("connection refused")):
            result = client.ensure_collection()

        assert result is False


class TestQdrantClientUpsert:
    """Test upsert method."""

    def test_success(self) -> None:
        client = QdrantClient()
        mock_qdrant_instance = MagicMock()
        mock_models = MagicMock()

        with (
            patch.object(client, "_get_client", return_value=mock_qdrant_instance),
            patch.dict("sys.modules", {"qdrant_client.models": mock_models}),
        ):
            result = client.upsert(
                ids=["id1", "id2"],
                vectors=[[0.1, 0.2], [0.3, 0.4]],
                payloads=[{"text": "hello"}, {"text": "world"}],
            )

        assert result is True
        mock_qdrant_instance.upsert.assert_called_once()

    def test_failure_returns_false(self) -> None:
        client = QdrantClient()
        with patch.object(client, "_get_client", side_effect=Exception("timeout")):
            result = client.upsert(
                ids=["id1"],
                vectors=[[0.1]],
                payloads=[{"text": "hello"}],
            )

        assert result is False


class TestQdrantClientSearch:
    """Test search method."""

    def test_success(self) -> None:
        client = QdrantClient()
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "hit1"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "result"}
        mock_qdrant.search.return_value = [mock_hit]

        with patch.object(client, "_get_client", return_value=mock_qdrant):
            results = client.search(query_vector=[0.1, 0.2], limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "hit1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["text"] == "result"

    def test_failure_returns_empty(self) -> None:
        client = QdrantClient()
        with patch.object(client, "_get_client", side_effect=Exception("unreachable")):
            results = client.search(query_vector=[0.1], limit=5)

        assert results == []

    def test_empty_payload_handled(self) -> None:
        client = QdrantClient()
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "hit1"
        mock_hit.score = 0.9
        mock_hit.payload = None
        mock_qdrant.search.return_value = [mock_hit]

        with patch.object(client, "_get_client", return_value=mock_qdrant):
            results = client.search(query_vector=[0.1], limit=1)

        assert results[0]["payload"] == {}


class TestQdrantClientDeleteCollection:
    """Test delete_collection method."""

    def test_success(self) -> None:
        client = QdrantClient()
        mock_qdrant = MagicMock()

        with patch.object(client, "_get_client", return_value=mock_qdrant):
            result = client.delete_collection()

        assert result is True
        mock_qdrant.delete_collection.assert_called_once()

    def test_failure_returns_false(self) -> None:
        client = QdrantClient()
        with patch.object(client, "_get_client", side_effect=Exception("error")):
            result = client.delete_collection()

        assert result is False
