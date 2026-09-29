import joblib
import numpy as np
import pytest

from stylometry.data.split import writer_split
from stylometry.explain.explain import coef_group_importance, local_contributions, top_features
from stylometry.models.train import fit_tuned


@pytest.fixture(scope="module")
def split(synthetic_df):
    return writer_split(synthetic_df, 0.25, seed=0)


@pytest.mark.parametrize("feature_set,model", [
    ("handcrafted", "lr"), ("char", "lr"), ("combined", "lr"), ("combined", "svm"),
    ("length", "lr"), ("none", "dummy"),
])
def test_fit_predict_proba(cfg, split, feature_set, model):
    train, test = split
    pipe, _, _ = fit_tuned(feature_set, model, train, cfg, seed=0)
    pred = pipe.predict(test.text)
    assert set(pred) <= {"native", "non_native"}
    proba = pipe.predict_proba(test.text)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_saved_model_gives_identical_predictions(cfg, split, tmp_path):
    train, test = split
    pipe, _, _ = fit_tuned("combined", "lr", train, cfg, seed=0)
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    again = joblib.load(path)
    assert (pipe.predict(test.text) == again.predict(test.text)).all()
    assert np.allclose(pipe.predict_proba(test.text), again.predict_proba(test.text))


def test_explanations(cfg, split):
    train, test = split
    pipe, _, _ = fit_tuned("combined", "lr", train, cfg, seed=0)
    top = top_features(pipe, k=5)
    assert set(top.direction) == {"native", "non_native"}
    loc = local_contributions(pipe, test.text.iloc[0], k=8)
    assert len(loc) <= 16
    assert set(loc.toward) <= {"native", "non_native"}
    svm, _, _ = fit_tuned("handcrafted", "svm", train, cfg, seed=0)
    gi = coef_group_importance(svm)
    assert set(gi.group) == {"sentence", "vocabulary", "function_words", "pos", "punctuation"}


def test_ablation_groups(cfg, split):
    train, test = split
    pipe, _, _ = fit_tuned("handcrafted", "lr", train, cfg, seed=0,
                           feature_groups=["sentence", "vocabulary"])
    names = pipe.named_steps["features"].get_feature_names_out()
    assert all(n.startswith(("sent_", "vocab_")) for n in names)
