"""Tests for the toy retrieval dataset integrity."""

import json
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).parent.parent / "data" / "toy_retrieval.json"


@pytest.fixture
def dataset():
    with open(DATA_PATH) as f:
        return json.load(f)


class TestDatasetSchema:
    def test_file_exists(self):
        assert DATA_PATH.exists()

    def test_has_documents_and_queries(self, dataset):
        assert "documents" in dataset
        assert "queries" in dataset

    def test_exactly_10_documents(self, dataset):
        assert len(dataset["documents"]) == 10

    def test_exactly_5_queries(self, dataset):
        assert len(dataset["queries"]) == 5

    def test_document_schema(self, dataset):
        for doc in dataset["documents"]:
            assert "id" in doc and isinstance(doc["id"], int)
            assert "text" in doc and isinstance(doc["text"], str)
            assert len(doc["text"]) > 0

    def test_query_schema(self, dataset):
        for q in dataset["queries"]:
            assert "id" in q and isinstance(q["id"], str)
            assert "text" in q and isinstance(q["text"], str)
            assert "relevant" in q and isinstance(q["relevant"], list)
            assert "hard_negative" in q and isinstance(q["hard_negative"], list)

    def test_document_ids_are_sequential(self, dataset):
        ids = [d["id"] for d in dataset["documents"]]
        assert ids == list(range(10))

    def test_relevant_ids_exist(self, dataset):
        doc_ids = {d["id"] for d in dataset["documents"]}
        for q in dataset["queries"]:
            for rid in q["relevant"]:
                assert rid in doc_ids, f"Query {q['id']} references non-existent doc {rid}"

    def test_hard_negative_ids_exist(self, dataset):
        doc_ids = {d["id"] for d in dataset["documents"]}
        for q in dataset["queries"]:
            for hid in q["hard_negative"]:
                assert hid in doc_ids, f"Query {q['id']} references non-existent doc {hid}"

    def test_no_overlap_relevant_and_hard_negative(self, dataset):
        for q in dataset["queries"]:
            overlap = set(q["relevant"]) & set(q["hard_negative"])
            assert len(overlap) == 0, f"Query {q['id']} has overlap: {overlap}"

    def test_q3_is_easy_control(self, dataset):
        q3 = next(q for q in dataset["queries"] if q["id"] == "q3")
        assert len(q3["relevant"]) == 1
        assert "control" in q3["note"].lower() or "straightforward" in q3["note"].lower()
