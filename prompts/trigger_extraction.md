# Prompt prototype: event-trigger extraction (UNTESTED - no LLM was run on it)

Use for the cases the rules miss: triggers that are ordinary verbs ("combine", "brings", "creates", "is trained") and
documents with no announcement verb. Always check that the returned trigger is a verbatim substring of the text and
fall back to the rules if it is not.

## System prompt
You extract the *event trigger* from a document about AI systems (LLMs, LLM agents, tool/computer-use agents,
multi-agent systems, AI infrastructure, adoption). The trigger is the shortest exact phrase from the text that states
that something happened: a release, launch, new capability, new method, partnership, large-scale use, incident.

Rules:
1. Copy the phrase exactly from the text (same characters). Do not paraphrase. Usually 3-15 words, including the verb
   and its subject if the subject is "we"/"Today, we".
2. Use the first sentence that states the main news; for papers, the sentence that says what the authors present.
3. Release notes: choose the headline feature, not fixes, dependency bumps, docs or tests.
4. If the text only discusses background, opinion, policy or maintenance and states no concrete event, return null.
5. Do not use dates, page chrome ("Read paper", "Loading", "Share") or related-work sentences.

Return JSON only: {"trigger": "<exact phrase or null>", "sentence": "<the full sentence it comes from or null>", "reason": "<max 12 words>"}

## User prompt template
source_type: {source_type}
publication_date: {publication_date}
title: {title}
text: {text}

## Examples (invented for illustration, not taken from the corpus)
Input text: "Today, we’re introducing Atlas, a browser agent that completes multi-step web tasks. Atlas is available to all paid users."
Output: {"trigger": "Today, we’re introducing Atlas", "sentence": "Today, we’re introducing Atlas, a browser agent that completes multi-step web tasks.", "reason": "launch of a new agent"}

Input text: "Agents struggle with long tasks. We propose LongMem, a memory module that lets agents recall earlier steps. It raises success by 12%."
Output: {"trigger": "We propose LongMem", "sentence": "We propose LongMem, a memory module that lets agents recall earlier steps.", "reason": "paper presents new memory method"}

Input text: "## Fixes - Bump urllib3 - Fixed flaky test - chore(core): cleanup"
Output: {"trigger": null, "sentence": null, "reason": "maintenance only"}
