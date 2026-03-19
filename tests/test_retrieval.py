from src.retriever import expand_with_xrefs


def test_xref_expansion_basic():
    chunks = [
        {"chunk_id": "a", "doc_id": "d", "page_start": 1, "task_id": "32-11", "section": "32-11-00", "figure_ids": [], "table_ids": [], "text": "Refer to TASK 32-11"},
        {"chunk_id": "b", "doc_id": "d", "page_start": 2, "task_id": "32-11", "section": "32-11-00", "figure_ids": [], "table_ids": [], "text": "Task details"},
    ]
    xrefs = [{"from_chunk_id": "a", "target_key": "32-11", "target_type": "task"}]
    out = expand_with_xrefs(["a"], chunks, xrefs, limit=1)
    assert out == ["b"]
