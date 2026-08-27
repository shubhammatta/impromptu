"""CLI flag parsing tests (no app launch)."""

import pytest

from prompt_enhancer.cli import build_parser


def test_defaults():
    args = build_parser().parse_args([])
    assert args.model  # from config
    assert args.host  # from config
    assert args.clarify is None  # app falls back to config default (off)
    assert args.level is None  # app falls back to DEFAULT_LEVEL (5)


def test_level_flag_accepted():
    for level in (1, 2, 3, 4, 5):
        assert build_parser().parse_args(["--level", str(level)]).level == level


def test_level_flag_rejects_out_of_range():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--level", "6"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--level", "0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--level", "polished"])


def test_clarify_flags():
    assert build_parser().parse_args(["--clarify"]).clarify is True
    assert build_parser().parse_args(["--no-clarify"]).clarify is False
