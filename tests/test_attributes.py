from ainevo.classification.attributes import rule_autonomy, rule_interaction


def test_rule_autonomy_levels():
    assert rule_autonomy({"title": "Agent", "text": "It works autonomously end-to-end."}) == "3"
    assert rule_autonomy({"title": "A benchmark", "text": "We release a dataset."}) == "NA"
    assert rule_autonomy({"title": "x", "text": "plain"}) == "0"


def test_rule_interaction():
    assert rule_interaction({"title": "Multi-agent framework", "text": "agents collaborate"}) == "agent_agent"
    assert rule_interaction({"title": "Chat model", "text": "a language model that answers prompts"}) == "human_model"
