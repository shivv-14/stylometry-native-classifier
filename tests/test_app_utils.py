import pandas as pd

from stylometry.app_utils import STYLE_PANEL, style_summary, validate_text


def test_refuses_empty():
    v = validate_text("   ")
    assert not v.ok and v.level == "error" and v.words == 0


def test_refuses_under_40_words():
    v = validate_text("word " * 39)
    assert not v.ok and v.level == "error" and v.words == 39


def test_warns_between_40_and_120():
    v = validate_text("word " * 40)
    assert v.ok and v.level == "warning"
    v = validate_text("word " * 119)
    assert v.ok and v.level == "warning"


def test_ok_from_120_words():
    v = validate_text("word " * 120)
    assert v.ok and v.level == "ok" and v.words == 120


def test_custom_thresholds():
    assert validate_text("a b c", min_words=2, warn_words=3).level == "ok"


def test_style_summary_uses_known_keys():
    row = pd.Series({k: 1.0 for k in STYLE_PANEL})
    out = style_summary(row)
    assert len(out) == len(STYLE_PANEL)
