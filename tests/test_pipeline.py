from ainevo.extraction.pipeline import extract_document

DOC = {"doc_id": "x1", "source": "OpenAI", "source_type": "lab_blog", "title": "Introducing GPT-4o",
       "publication_date": "2024-05-13",
       "text": "Today we are introducing GPT-4o, our new flagship model that can reason across audio, vision and text."}


def test_extract_document_fields():
    r = extract_document(DOC)
    assert r["has_candidate_event"] and r["trigger"]
    assert r["event_date"] == "2024-05-13" and r["event_date_how"] == "publication_date"
    assert r["organizations"] == ["OpenAI"] and r["systems"][0] == "GPT-4o"


def test_empty_text_does_not_crash():
    r = extract_document({**DOC, "text": ""})
    assert r["n_units"] == 0 and r["trigger"] is None
