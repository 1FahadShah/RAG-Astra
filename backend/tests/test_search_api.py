
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.services.semantic_retrieval import RetrievalInputError


class TestHybridSearchAPI(unittest.TestCase):
    def setUp(self):
        def override_get_db():
            yield None

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)

    @staticmethod
    def sample_result():
        return {
            "chunk_id": uuid.UUID(int=1),
            "document_id": uuid.UUID(int=1001),
            "filename": "security-handbook.txt",
            "chunk_index": 0,
            "text": "Security controls protect enterprise data.",
            "page_number": 7,
            "section_title": "Security",
            "semantic_rank": 1,
            "keyword_rank": 2,
            "similarity_score": 0.91,
            "keyword_score": 0.08,
            "rrf_score": 1 / 61 + 1 / 62,
        }

    @patch("app.api.search.retrieve_hybrid_chunks")
    def test_successful_search_returns_ranked_results(self, mock_search):
        mock_search.return_value = [self.sample_result()]

        response = self.client.post(
            "/api/v1/search",
            json={"query": "  security controls  "},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["query"], "security controls")
        self.assertEqual(payload["limit"], 5)
        self.assertEqual(payload["candidate_limit"], 20)
        self.assertEqual(payload["rrf_k"], 60)
        self.assertEqual(payload["returned"], 1)

        item = payload["items"][0]
        self.assertEqual(item["filename"], "security-handbook.txt")
        self.assertEqual(item["page_number"], 7)
        self.assertEqual(item["semantic_rank"], 1)
        self.assertEqual(item["keyword_rank"], 2)

        mock_search.assert_called_once_with(
            db=None,
            query="security controls",
            limit=5,
            candidate_limit=None,
            rrf_k=60,
        )

    @patch("app.api.search.retrieve_hybrid_chunks")
    def test_custom_retrieval_settings_are_forwarded(self, mock_search):
        mock_search.return_value = []

        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "enterprise security",
                "limit": 3,
                "candidate_limit": 7,
                "rrf_k": 40,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["candidate_limit"], 7)
        self.assertEqual(payload["rrf_k"], 40)
        self.assertEqual(payload["returned"], 0)

        mock_search.assert_called_once_with(
            db=None,
            query="enterprise security",
            limit=3,
            candidate_limit=7,
            rrf_k=40,
        )

    @patch("app.api.search.retrieve_hybrid_chunks")
    def test_empty_results_return_success(self, mock_search):
        mock_search.return_value = []

        response = self.client.post(
            "/api/v1/search",
            json={"query": "no matching content"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])
        self.assertEqual(response.json()["returned"], 0)

    def test_invalid_requests_return_422(self):
        invalid_payloads = [
            {},
            {"query": ""},
            {"query": "   "},
            {"query": "valid", "limit": 0},
            {"query": "valid", "limit": 101},
            {"query": "valid", "candidate_limit": 0},
            {"query": "valid", "candidate_limit": 101},
            {"query": "valid", "rrf_k": 0},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/api/v1/search",
                    json=payload,
                )
                self.assertEqual(response.status_code, 422)

    @patch("app.api.search.retrieve_hybrid_chunks")
    def test_retrieval_input_error_returns_422(self, mock_search):
        mock_search.side_effect = RetrievalInputError(
            "The search query must be non-empty."
        )

        response = self.client.post(
            "/api/v1/search",
            json={"query": "valid query"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn(
            "non-empty",
            response.json()["detail"],
        )

    @patch("app.api.search.retrieve_hybrid_chunks")
    def test_unexpected_error_returns_safe_500(self, mock_search):
        mock_search.side_effect = RuntimeError("internal detail")

        response = self.client.post(
            "/api/v1/search",
            json={"query": "valid query"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "An internal error occurred during hybrid search.",
        )
        self.assertNotIn(
            "internal detail",
            response.text,
        )


if __name__ == "__main__":
    unittest.main()
