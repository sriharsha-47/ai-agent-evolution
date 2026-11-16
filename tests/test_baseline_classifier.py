from ainevo.classification.baseline import doc_text, make_model


def test_doc_text_uses_title_and_source_only():
    d = {"source_type": "paper", "title": "ATLAS: trading agents", "text": "x" * 5000, "category": "SECRET"}
    t = doc_text(d)
    assert t.startswith("paper ATLAS") and "SECRET" not in t and len(t) < 3100


def test_model_fits_and_predicts():
    X = ["agents coordinate and delegate tasks", "multi agent communication protocol", "agents collaborate in teams",
         "new model improves reasoning benchmark", "model release stronger coding ability", "improved reasoning model"] * 2
    y = ["multi_agent_coordination"] * 3 + ["capability_breakthrough"] * 3
    y = y * 2
    m = make_model().fit(X, y)
    assert m.predict(["agents collaborate"])[0] == "multi_agent_coordination"
