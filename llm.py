import re
import logging

import litellm

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


def call_llm(system_prompt: str, user_prompt: str, model: str, temperature: float = 0.3) -> str:
    """Call LLM and return the response text."""
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def extract_code(response: str, language: str = "cpp") -> str | None:
    """Extract code block from LLM response. Returns None if no code found."""
    # Try language-specific fence first, then generic
    lang_aliases = {"cpp": ["cpp", "c++", "cxx"], "c": ["c"], "python": ["python", "py"]}
    aliases = lang_aliases.get(language, [language])
    for alias in aliases:
        match = re.search(rf'```{alias}\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
    # Fallback: generic code fence
    match = re.search(r'```\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
