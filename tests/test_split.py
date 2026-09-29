import pandas as pd
import pytest

from stylometry.data.balance import balance_writers
from stylometry.data.loaders import load_csv, load_wi_locness, parse_icnale_filename
from stylometry.data.split import (WriterLeakageError, assert_no_writer_overlap, class_ratio, topic_split,
                                   writer_split)

from .conftest import FIXTURE


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_writer_split_no_overlap(synthetic_df, seed):
    train, test = writer_split(synthetic_df, 0.2, seed)
    assert not set(train.writer_id) & set(test.writer_id)
    assert len(train) + len(test) == len(synthetic_df)


def test_class_ratio_preserved(synthetic_df):
    train, test = writer_split(synthetic_df, 0.2, 0)
    overall = class_ratio(synthetic_df)
    assert abs(class_ratio(train) - overall) < 0.15
    assert abs(class_ratio(test) - overall) < 0.15


def test_overlap_is_detected():
    a = pd.DataFrame({"writer_id": ["w1", "w2"]})
    b = pd.DataFrame({"writer_id": ["w2", "w3"]})
    with pytest.raises(WriterLeakageError):
        assert_no_writer_overlap(a, b)


def test_topic_split_unseen_writers(synthetic_df):
    train, test = topic_split(synthetic_df, "PTJ", seed=0)
    assert set(train.topic) == {"PTJ"}
    assert "PTJ" not in set(test.topic)
    assert not set(train.writer_id) & set(test.writer_id)


def test_balance_writers_equalises_writer_counts(synthetic_df):
    extra = synthetic_df[synthetic_df.label == "non_native"].copy()
    extra["writer_id"] = extra.writer_id + "_x"
    df = pd.concat([synthetic_df, extra], ignore_index=True)
    out = balance_writers(df, seed=0, strata=["l1"])
    counts = out.groupby("label").writer_id.nunique()
    assert counts["native"] == counts["non_native"]
    # every text of a selected writer is kept
    kept = out.writer_id.unique()
    assert len(out) == int(df.writer_id.isin(kept).sum())


def test_icnale_filename_parser():
    pattern = r'^W_(?P<country>[A-Z]{3})_(?P<topic>[A-Z]{3})\d*_(?P<writer>\d+)_(?P<proficiency>[A-Z0-9]+_\d+)(?:_[A-Za-z0-9]+)?\.txt$'
    m = parse_icnale_filename("W_CHN_PTJ0_004_B1_1.txt", pattern)
    assert m == {"writer_id": "CHN_004", "l1": "CHN", "topic": "PTJ", "proficiency": "B1_1", "label": "non_native"}
    assert parse_icnale_filename("W_ENS_SMK0_004_XX_1.txt", pattern)["label"] == "native"
    with pytest.raises(ValueError, match="does not match"):
        parse_icnale_filename("essay_12.txt", pattern)


def test_load_csv_fixture():
    df = load_csv(FIXTURE)
    assert {"text", "label", "writer_id", "topic", "l1", "proficiency"} <= set(df.columns)


def test_balance_keeps_both_classes_when_already_equal(synthetic_df):
    out = balance_writers(synthetic_df, seed=0, strata=["l1"])
    assert set(out.label) == {"native", "non_native"}
    assert len(out) == len(synthetic_df)


def test_wi_locness_loader(tmp_path):
    import json
    (tmp_path / "json").mkdir()
    learner = [{"text": "SYNTHETIC one", "id": "1-1", "userid": "u1", "cefr": "B1.ii", "edits": []},
               {"text": "SYNTHETIC two", "id": "1-2", "userid": "u1", "cefr": "B1.ii", "edits": []},
               {"text": "SYNTHETIC no writer", "id": "1-3", "userid": None, "cefr": "A2.i", "edits": []}]
    native = [{"text": "SYNTHETIC three", "id": "7-1", "cefr": "N", "edits": []}]
    (tmp_path / "json" / "B.train.json").write_text("\n".join(map(json.dumps, learner)), encoding="utf-8")
    (tmp_path / "json" / "N.dev.json").write_text("\n".join(map(json.dumps, native)), encoding="utf-8")
    df = load_wi_locness(tmp_path)
    assert sorted(df.label) == ["native", "non_native", "non_native"]
    assert df[df.label == "non_native"].writer_id.nunique() == 1
    assert set(df[df.label == "non_native"].proficiency) == {"B1"}
