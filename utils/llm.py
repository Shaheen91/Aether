"""
utils/llm.py

All Groq LLM integration lives here: building prompts from detected
diseases and calling the Groq chat completion API to generate a
disease explanation, treatment plan, and prevention advice.
"""

import logging

from groq import Groq

import config

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _build_user_prompt(detected_diseases: list[str], confidence_scores: list[float]) -> str:
    if not detected_diseases:
        return """The plant disease detection system analyzed an apple leaf and found NO diseases.
The leaf appears completely healthy.

Please provide:
- Confirmation of what a healthy apple leaf looks like
- Preventive care tips to keep it healthy
- Early warning signs the farmer should watch for in the future
- Ideal conditions for apple plant health

Keep it encouraging and practical."""

    if detected_diseases == ["complex"]:
        return f"""The plant disease detection system detected: COMPLEX infection (confidence: {confidence_scores[0]:.1%})

This means the leaf shows multiple overlapping symptoms that couldn't be cleanly separated into individual diseases.

Please provide:
- What complex infection means for the plant
- The most likely combination of diseases causing this
- Why this is serious and requires immediate attention
- Aggressive treatment protocol for multi-disease infection
- When to consider removing affected leaves entirely"""

    if len(detected_diseases) == 1:
        d = detected_diseases[0]
        context = config.DISEASE_KNOWLEDGE.get(d, d)
        return f"""The plant disease detection system detected a single disease on an apple leaf:

Disease: {d.replace('_', ' ').title()}
Confidence: {confidence_scores[0]:.1%}
Background: {context}

Provide a complete analysis and treatment guide for this specific disease."""

    diseases_str = "\n".join(
        f"- {d.replace('_', ' ').title()}: {config.DISEASE_KNOWLEDGE.get(d, d)} "
        f"(confidence: {c:.1%})"
        for d, c in zip(detected_diseases, confidence_scores)
    )
    return f"""The plant disease detection system detected MULTIPLE diseases on the same apple leaf:

{diseases_str}

This is a multi-disease infection. Please:
- Explain each disease separately first
- Then explain how they interact and why co-infection is more serious
- Provide a combined treatment protocol that addresses all detected diseases
- Prioritize which disease to treat first if resources are limited"""


def get_disease_explanation(
    detected_diseases: list[str], confidence_scores: list[float]
) -> str:
    """
    Calls the Groq LLM to generate a plant pathologist style explanation
    covering diagnosis, spread, immediate actions, treatment, and the
    consequences of ignoring the issue.
    """
    client = _get_client()
    user_prompt = _build_user_prompt(detected_diseases, confidence_scores)

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.exception("Groq LLM call failed")
        raise RuntimeError(f"Failed to generate explanation: {exc}") from exc
