
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from app.services.hybrid_retrieval import (
    DEFAULT_RRF_K,
    retrieve_hybrid_chunks,
)
from app.services.semantic_retrieval import (
    MAX_RETRIEVAL_LIMIT,
    RetrievalInputError,
)


def make_result(
    number,
    *,
    similarity_score=None,
    keyword_score=None,
):
    return SimpleNamespace(
        chunk_id=uuid.UUID(int=number),
        document_id=uuid.UUID(int=number + 1000),
        filename=f"document-{number}.txt",
        chunk_index=0,
        text=f"Text for chunk {number}",
        page_number=number,
        section_title="Test section",
        similarity_score=similarity_score,
        keyword_score=keyword_score,
    )


class TestHybridRetrieval(unittest.TestCase):
    def run_hybrid(
        self,
        semantic_results,
        keyword_results,
        **kwargs,
    ):
        with (
            patch(
                "app.services.hybrid_retrieval.retrieve_chunks",
                return_value=semantic_results,
            ) as semantic_mock,
            patch(
                "app.services.hybrid_retrieval.retrieve_keyword_chunks",
                return_value=keyword_results,
            ) as keyword_mock,
        ):
            results = retrieve_hybrid_chunks(
                db=None,
                query="  example query  ",
                **kwargs,
            )

        return results, semantic_mock, keyword_mock

    def test_fuses_both_rankings_and_deduplicates_overlap(self):
        semantic = [
            make_result(1, similarity_score=0.99),
            make_result(2, similarity_score=0.90),
        ]
        keyword = [
            make_result(2, keyword_score=0.80),
            make_result(3, keyword_score=0.70),
        ]

        results, _, _ = self.run_hybrid(semantic, keyword)

        self.assertEqual(
            [result.chunk_id for result in results],
            [
                uuid.UUID(int=2),
                uuid.UUID(int=1),
                uuid.UUID(int=3),
            ],
        )

        shared = results[0]
        self.assertEqual(shared.semantic_rank, 2)
        self.assertEqual(shared.keyword_rank, 1)
        self.assertAlmostEqual(
            shared.rrf_score,
            1 / (DEFAULT_RRF_K + 2)
            + 1 / (DEFAULT_RRF_K + 1),
        )
        self.assertEqual(shared.filename, "document-2.txt")
        self.assertEqual(shared.page_number, 2)

    def test_single_source_results_have_optional_missing_fields(self):
        results, _, _ = self.run_hybrid(
            [make_result(1, similarity_score=0.95)],
            [make_result(2, keyword_score=0.50)],
        )

        semantic_only = next(
            result for result in results
            if result.chunk_id == uuid.UUID(int=1)
        )
        keyword_only = next(
            result for result in results
            if result.chunk_id == uuid.UUID(int=2)
        )

        self.assertEqual(semantic_only.semantic_rank, 1)
        self.assertIsNone(semantic_only.keyword_rank)
        self.assertIsNone(semantic_only.keyword_score)

        self.assertIsNone(keyword_only.semantic_rank)
        self.assertEqual(keyword_only.keyword_rank, 1)
        self.assertIsNone(keyword_only.similarity_score)

    def test_ranking_uses_rank_positions_not_raw_scores(self):
        semantic = [
            make_result(1, similarity_score=0.99),
            make_result(2, similarity_score=0.90),
        ]
        keyword = [
            make_result(2, keyword_score=0.01),
            make_result(1, keyword_score=0.99),
        ]

        results, _, _ = self.run_hybrid(semantic, keyword)

        # Both chunks occur at ranks 1 and 2, so their RRF scores tie.
        # Deterministic UUID ordering breaks the tie.
        self.assertEqual(
            [result.chunk_id for result in results],
            [uuid.UUID(int=1), uuid.UUID(int=2)],
        )
        self.assertAlmostEqual(
            results[0].rrf_score,
            results[1].rrf_score,
        )

    def test_candidate_limit_defaults_to_four_times_output_limit(self):
        _, semantic_mock, keyword_mock = self.run_hybrid(
            [],
            [],
            limit=5,
        )

        semantic_mock.assert_called_once_with(
            None,
            "example query",
            limit=20,
        )
        keyword_mock.assert_called_once_with(
            None,
            "example query",
            limit=20,
        )

    def test_candidate_limit_is_capped_at_maximum(self):
        _, semantic_mock, keyword_mock = self.run_hybrid(
            [],
            [],
            limit=MAX_RETRIEVAL_LIMIT,
        )

        self.assertEqual(
            semantic_mock.call_args.kwargs["limit"],
            MAX_RETRIEVAL_LIMIT,
        )
        self.assertEqual(
            keyword_mock.call_args.kwargs["limit"],
            MAX_RETRIEVAL_LIMIT,
        )

    def test_explicit_candidate_limit_is_used(self):
        _, semantic_mock, keyword_mock = self.run_hybrid(
            [],
            [],
            limit=3,
            candidate_limit=7,
        )

        self.assertEqual(
            semantic_mock.call_args.kwargs["limit"],
            7,
        )
        self.assertEqual(
            keyword_mock.call_args.kwargs["limit"],
            7,
        )

    def test_output_limit_is_respected(self):
        semantic = [
            make_result(index, similarity_score=0.9)
            for index in range(1, 5)
        ]

        results, _, _ = self.run_hybrid(
            semantic,
            [],
            limit=2,
        )

        self.assertEqual(len(results), 2)

    def test_empty_retrieval_lists_return_empty_results(self):
        results, _, _ = self.run_hybrid([], [])

        self.assertEqual(results, [])

    def test_invalid_query_limit_candidate_limit_and_rrf_k(self):
        invalid_arguments = [
            {"query": "", "limit": 5},
            {"query": "   ", "limit": 5},
            {"query": "valid", "limit": 0},
            {"query": "valid", "limit": MAX_RETRIEVAL_LIMIT + 1},
            {"query": "valid", "limit": True},
            {"query": "valid", "limit": 5, "candidate_limit": 0},
            {
                "query": "valid",
                "limit": 5,
                "candidate_limit": MAX_RETRIEVAL_LIMIT + 1,
            },
            {"query": "valid", "limit": 5, "candidate_limit": True},
            {"query": "valid", "limit": 5, "rrf_k": 0},
            {"query": "valid", "limit": 5, "rrf_k": True},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(RetrievalInputError):
                    retrieve_hybrid_chunks(db=None, **arguments)


if __name__ == "__main__":
    unittest.main()
