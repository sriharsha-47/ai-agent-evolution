import json

from ainevo.storage import db


def rows():
    return [
        {"event_id": "GE-0001", "annotation_id": "EX-069", "doc_id": "d_keep", "source_url": "http://a"},
        {"event_id": "GE-0002", "annotation_id": "EX-071", "doc_id": "d_drop", "source_url": "http://b"},
        {"event_id": "GE-0003", "annotation_id": "EX-072", "doc_id": "d_other", "source_url": "http://c"},
    ]


def test_merge_keeps_one_event_and_both_documents():
    out, msgs = db.apply_gold_merges(rows(), [("EX-069", "EX-071")])
    assert [r["annotation_id"] for r in out] == ["EX-069", "EX-072"]
    srcs = json.loads(out[0]["sources"])
    assert [s["doc_id"] for s in srcs] == ["d_keep", "d_drop"]
    assert msgs == ["merged EX-071 into EX-069"]


def test_merge_is_found_by_annotation_id_not_event_id():
    r = rows()
    r[0]["event_id"], r[1]["event_id"] = "GE-0900", "GE-0777"      # renumbered ids
    out, _ = db.apply_gold_merges(r, [("EX-069", "EX-071")])
    assert len(out) == 2


def test_missing_annotation_id_is_skipped_with_a_message():
    out, msgs = db.apply_gold_merges(rows(), [("EX-069", "EX-999")])
    assert len(out) == 3 and "skipped" in msgs[0]


def test_read_merges_handles_bom_and_missing_file(tmp_path):
    f = tmp_path / "m.csv"
    f.write_bytes("\ufeffkeep_annotation_id,drop_annotation_id,reason\r\nEX-069,EX-071,x\r\n".encode("utf-8"))
    assert db.read_gold_merges(f) == [("EX-069", "EX-071")]
    assert db.read_gold_merges(tmp_path / "none.csv") == []
