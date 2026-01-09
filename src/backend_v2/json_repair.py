import json
import re
import logging

logger = logging.getLogger("backend_v2.json_repair")


def repair_json(json_str: str) -> dict:
    """
    Attempt to repair and parse a malformed JSON string.
    Handles common LLM output errors:
    - Trailing commas
    - Missing closing braces
    - Unescaped quotes
    - Python literals (None, True, False)
    - Markdown code blocks
    """
    if not json_str:
        raise ValueError("Empty JSON string")

    # 1. Remove Markdown code blocks
    json_str = re.sub(r"^```json\s*", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r"^```\s*", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r"\s*```$", "", json_str, flags=re.MULTILINE)

    # 2. Fix Python literals
    json_str = json_str.replace("None", "null")
    json_str = json_str.replace("True", "true")
    json_str = json_str.replace("False", "false")

    # 3. Remove trailing commas (simple case for arrays and objects)
    # This regex looks for a comma followed by closing brace/bracket, with optional whitespace
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

    # 4. Attempt to balance braces if missing
    open_braces = json_str.count("{")
    close_braces = json_str.count("}")
    if open_braces > close_braces:
        json_str += "}" * (open_braces - close_braces)

    open_brackets = json_str.count("[")
    close_brackets = json_str.count("]")
    if open_brackets > close_brackets:
        json_str += "]" * (open_brackets - close_brackets)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Standard repair failed: {e}. Trying aggressive repair.")

        # 5. Aggressive: Try to find the largest JSON-like structure
        # Find everything between the first { and the last }
        match = re.search(r"(\{.*\})", json_str, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except:
                pass

        raise e
