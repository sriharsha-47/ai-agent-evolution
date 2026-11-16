from ainevo.extraction.relations import action_phrase, extract_relation, rule_capability

DOC = {"doc_id": "d1", "source": "OpenAI", "source_type": "lab_blog", "title": "Introducing GPT-4o", "publication_date": "2024-05-13",
       "text": "Today we are introducing GPT-4o, a model that can reason across audio, vision and text in real time."}


def test_rule_capability():
    assert rule_capability({"title": "Multi-agent framework", "text": "agents collaborate and delegate"}) == "multi_agent_coordination"
    assert rule_capability({"title": "x", "text": "nothing relevant here"}) == "other"
    assert rule_capability(DOC) == "multimodal"


def test_action_phrase_trims_to_clause():
    s = "Today we are introducing GPT-4o, a model that can reason across audio, vision and text in real time and more words here."
    a = action_phrase(s, "introducing")
    assert a.startswith("introducing GPT-4o") and len(a.split()) <= 22


def test_extract_relation_keys():
    r = extract_relation(DOC)
    assert r["organization"] == ["OpenAI"] and r["system"][0] == "GPT-4o" and r["capability"] and r["action"]
