from prompt_enhancer.text import extract_prompt


def test_plain_text_passthrough():
    raw = "# Role\nYou are a grumpy chef."
    assert extract_prompt(raw) == raw


def test_strips_wrapping_fence():
    raw = "```markdown\n# Role\nYou are a grumpy chef.\n```"
    assert extract_prompt(raw) == "# Role\nYou are a grumpy chef."


def test_strips_fence_with_language_and_trailing_newline():
    raw = "```md\nBody line 1\nBody line 2\n```\n"
    assert extract_prompt(raw) == "Body line 1\nBody line 2"


def test_strips_think_block_then_fence():
    raw = "<think>The user wants a chef persona.</think>\n```text\nBe a chef.\n```"
    assert extract_prompt(raw) == "Be a chef."


def test_strips_unterminated_think_block():
    assert extract_prompt("<think>reasoning without a closing tag") == ""


def test_leaves_non_wrapping_fence_alone():
    raw = "Use this:\n```py\nx = 1\n```\nEnjoy."
    assert extract_prompt(raw) == raw
