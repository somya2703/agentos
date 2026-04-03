"""
agents/coder.py — Coder sub-agent
"""
import logging
import re
from agents.base_agent import BaseAgent
from tools.code_executor import execute_python

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    agent_id = "coder"
    description = (
        "You write clean, well-commented Python code to solve programming tasks. "
        "You think step-by-step about the solution before writing code. "
        "You always test your code mentally and flag potential edge cases."
    )

    def run(self, task: str, context: str = "", **kwargs) -> str:
        ctx = f"\nContext:\n{context}" if context else ""
        prompt = (
            f"Programming task: {task}{ctx}\n\n"
            "Write a clean Python solution. Include:\n"
            "1. A brief explanation of your approach\n"
            "2. The complete Python code in a ```python block\n"
            "3. Example output or usage\n"
        )
        response = self.think_fresh(prompt)

        # Optionally execute the code
        code = _extract_code(response)
        if code:
            try:
                exec_result = self.use_tool("code_executor", execute_python, code=code)
                if exec_result.get("success"):
                    response += f"\n\n**Execution output:**\n```\n{exec_result['output']}\n```"
                else:
                    response += f"\n\n**Execution error:** {exec_result.get('error')}"
            except Exception as e:
                logger.warning(f"Code execution skipped: {e}")

        self.log_action("code", task, response)
        return response


def _extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""
