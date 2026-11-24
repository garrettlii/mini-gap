"""Free-form code agent (baseline): instruction -> Python policy over the same skills.

Sees exactly what the planner sees (workcell, catalog, one worked example), as code.
"""

from __future__ import annotations

import ast
from pathlib import Path

from gap.agents.context import CODE_RULES, WORKCELL, fill
from gap.agents.llm import LLM
from gap.skills import catalog_text

PROMPT = (Path(__file__).parent / "prompts" / "coder.md").read_text()


def write_code(llm: LLM, instruction: str, max_attempts: int = 3) -> tuple[str | None, list[str]]:
    """Return (code, attempt_log). Syntax and structure errors go back to the model."""
    system = fill(PROMPT, catalog=catalog_text(), workcell=WORKCELL, code_rules=CODE_RULES,
                  instruction=instruction)
    messages = [{"role": "user", "content": f"Task: {instruction}\nOutput the code."}]
    log: list[str] = []
    for _ in range(max_attempts):
        text = llm.complete(system, messages, max_tokens=1500).text
        code = extract_code(text)
        problem = check_code(code)
        log.append(problem or "valid")
        if problem is None:
            return code, log
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That code has a problem: {problem}\n"
                                                 "Fix it and output the whole code again."}]
    return None, log


def extract_code(text: str) -> str:
    if "```" not in text:
        return text.strip()
    block = text.split("```")[1]
    return block.removeprefix("python").strip()


def check_code(code: str) -> str | None:
    """None if the code parses, defines run(), and imports nothing; else the problem."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error on line {e.lineno}: {e.msg}"
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
        return "imports are not allowed; call the skills directly"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        return "define a function run() at the top level"
    return None
