from ainevo.extraction.ner import owner_from_source, title_system_name, guess_organizations


def test_owner_from_source():
    assert owner_from_source("OpenAI", "lab_blog") == "OpenAI"
    assert owner_from_source("microsoft/autogen", "release_notes") == "microsoft"
    assert owner_from_source("arXiv", "paper") is None


def test_title_system_name():
    assert title_system_name("ATLAS: Adaptive Trading with LLM AgentS", "paper") == "ATLAS"
    assert title_system_name("Scalable and Safe Multi-Agent Planning", "paper") is None
    assert title_system_name("crewAIInc/crewAI 1.8.1", "release_notes") == "crewAI 1.8.1"
    assert title_system_name("Introducing GPT-4o and more tools to ChatGPT", "lab_blog") == "GPT-4o"


def test_guess_orgs_adds_title_partner():
    d = {"source": "OpenAI", "source_type": "lab_blog", "title": "OpenAI and Amazon announce partnership"}
    assert guess_organizations(d) == ["OpenAI", "Amazon"]


def test_hyphen_canon_and_o1_family():
    from ainevo.extraction.ner import canon, guess_systems
    assert canon("GPT‑4o") == "GPT-4o"
    d = {"source": "OpenAI", "source_type": "lab_blog", "title": "Coding with OpenAI o1", "text": "A customer story about ChatGPT."}
    assert guess_systems(d)[0] == "o1"
    d2 = {"source": "OpenAI", "source_type": "lab_blog", "title": "OpenAI o1‑mini", "text": "x"}
    assert guess_systems(d2)[0] == "o1-mini"
    d3 = {"source": "OpenAI", "source_type": "lab_blog", "title": "Introducing GPT‑4o", "text": "x"}
    assert guess_systems(d3)[0] == "GPT-4o"
