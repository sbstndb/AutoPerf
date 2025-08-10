import os

try:
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_core.runnables import RunnableLambda  # type: ignore
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore
except (ImportError, ModuleNotFoundError):  # Optional dependency; runtime will fail earlier if not installed
    ChatPromptTemplate = None  # type: ignore
    RunnableLambda = None  # type: ignore
    StrOutputParser = None  # type: ignore
    ChatOpenAI = None  # type: ignore

from orchestrator.src.config import PROMPTS_DIR
from orchestrator.src.utils import extract_code_from_markdown





def create_langchain_optimizer(
    model,
    temperature,
    prompt_file,
    prompt_files,
    thinking_config=None,
):
    if ChatPromptTemplate is None or RunnableLambda is None or StrOutputParser is None or ChatOpenAI is None:
        raise RuntimeError("LangChain/OpenAI packages are required for optimizer but not installed.")
    # Import du thinking config si fourni
    from orchestrator.src.thinking_config import ThinkingConfig, ThinkingMode
    
    if thinking_config is None:
        thinking_config = ThinkingConfig(mode=ThinkingMode.DISABLED)
    
    thinking_instructions = thinking_config.get_thinking_instructions()
    
    if thinking_config.mode == ThinkingMode.DISABLED:
        user_prompt_template = """You are an expert C++ programmer. Improve the performance of the code below.

Previous ns/item: {score:.2f}. Try to reduce it.
Keep exact behavior. Do not change the interface.

IMPORTANT: Return ONLY the full C++ code for the kernel.
Do not add any comments, explanations, or markdown formatting.
Your output must be only the raw C++ code.

===== current kernel.cpp =====
{kernel_code}
"""
    else:
        user_prompt_template = f"""You are an expert C++ programmer. Improve the performance of the code below.

Previous ns/item: {{score:.2f}}. Try to reduce it.
Keep exact behavior. Do not change the interface.

{thinking_instructions}

IMPORTANT: After your thinking (if any), return ONLY the full C++ code for the kernel.
Do not add any comments, explanations, or markdown formatting outside of thinking tags.
Your final output must be only the raw C++ code.

===== current kernel.cpp =====
{{kernel_code}}
"""

    if prompt_files:
        prompts = []
        for p in prompt_files:
            with open(p, 'r', encoding='utf-8') as f:
                prompts.append(f.read())

        def select_prompt(inputs):
            phase = inputs.get("phase", 0)
            system_prompt = prompts[phase % len(prompts)]
            return ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("user", user_prompt_template)]
            )

        prompt_selector = RunnableLambda(select_prompt)
    elif prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt_template)]
        )
    else:
        with open(os.path.join(PROMPTS_DIR, "system.txt"), 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt_template)]
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    if prompt_files:
        return prompt_selector | llm | StrOutputParser() | extract_code_from_markdown
    else:
        return prompt | llm | StrOutputParser() | extract_code_from_markdown
