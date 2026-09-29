"""Streamlit demo: "Who Wrote This?" - native vs non-native English writing.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from stylometry.app_utils import style_summary, validate_text  # noqa: E402
from stylometry.config import load_config, resolve  # noqa: E402
from stylometry.explain.explain import coef_group_importance, local_contributions  # noqa: E402
from stylometry.features.handcrafted import HandcraftedFeatures  # noqa: E402

DISCLAIMER = (
    "**Experimental research demo.** Output is a model confidence based on patterns in one dataset, "
    "not a judgment about anyone's identity, nationality, intelligence, or English ability. "
    "Do not use for academic, employment, immigration, or disciplinary decisions."
)

# Neutral example passages written for this demo (not real people's text).
EXAMPLES = {
    "Example 1 - part-time jobs": (
        "Many university students take part-time jobs while they study. A job can give them some money "
        "for books, rent and food, and it can also show them how a workplace is organised. They learn to "
        "arrive on time, to talk politely with customers, and to work in a team. On the other hand, a job "
        "takes time and energy. A student who works every evening may be too tired to read or to prepare "
        "for exams, and grades can suffer. For this reason I think the number of hours matters more than "
        "the job itself. Ten hours a week is usually manageable, but thirty hours is a second full-time "
        "life. Universities could help by offering flexible jobs on campus, such as working in the library "
        "or helping in a laboratory, so that students gain experience without losing the purpose of their "
        "studies. In the end, each student has to find a balance that fits their own situation and goals."
    ),
    "Example 2 - smoking in restaurants": (
        "Should smoking be banned in all restaurants? In my opinion, the answer is yes. A restaurant is a "
        "shared space, and the smoke from one table quickly reaches the others. People who do not smoke, "
        "including children and staff, have no real choice about breathing it. Staff are especially "
        "affected, because they spend many hours a day in the same room. Some owners worry that a ban will "
        "reduce the number of customers. However, several cities that introduced such rules found that "
        "most restaurants kept their customers, and some even gained new ones who had avoided smoky places "
        "before. Smokers can still go outside for a few minutes if they wish. A clear national rule is also "
        "fairer than leaving the decision to each owner, since it treats every business in the same way. "
        "For these reasons, a complete ban seems like a reasonable step for public health."
    ),
    "Example 3 - learning a language": (
        "Learning a new language takes a long time, and there is no single method that works for everyone. "
        "Some people like to study grammar rules first, while others prefer to listen to music, watch films "
        "and copy what they hear. Both approaches have value. Grammar gives a clear structure, so the "
        "learner can build correct sentences, but it can feel slow and dry. Listening and speaking are more "
        "enjoyable, yet mistakes may become habits if nobody corrects them. A mix of the two is probably "
        "the best choice. For example, a learner could spend a short time each day on one grammar point "
        "and then use it in a conversation with a friend or a teacher. Regular practice matters more than "
        "long sessions once a week. Small steps, repeated often, slowly turn into real confidence when "
        "speaking and writing."
    ),
}

st.set_page_config(page_title="Who Wrote This?", page_icon="✍️", layout="wide")
cfg = load_config()
models_dir = resolve(cfg, "models_dir")
results_dir = resolve(cfg, "results_dir")


@st.cache_resource
def load_model(name: str):
    return joblib.load(models_dir / f"{name}.joblib")


@st.cache_data
def load_metadata():
    p = models_dir / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_resource
def style_extractor():
    f = cfg["features"]
    return HandcraftedFeatures(function_words=tuple(f["function_words"]), pos_tags=tuple(f["pos_tags"]),
                               ttr_window=f["ttr_window"], mattr_window=f["mattr_window"],
                               long_word_len=f["long_word_len"], spacy_model=f["spacy_model"]).fit([])


st.title("Who Wrote This?")
st.caption("Detecting native and non-native English writing with stylometric features "
           "(sentence, vocabulary, function-word, POS, punctuation) and character n-grams.")
st.warning(DISCLAIMER)

meta = load_metadata()
if meta is None:
    st.error("No trained model found in `models/`. Obtain the corpus (see `data/README.md`), then run "
             "`python tasks.py prepare train evaluate`.")
    st.stop()

# ---- sidebar -------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    model_label = st.radio("Classifier", ["Logistic Regression", "Linear SVM (calibrated)"])
    model_key = "lr" if model_label.startswith("Logistic") else "svm"
    sets = sorted({m["feature_set"] for m in meta["models"].values()})
    best = meta["best_feature_set"]
    feature_set = st.selectbox("Feature set", sets, index=sets.index(best),
                               format_func=lambda s: f"{s} (best)" if s == best else s)
    st.markdown(f"[GitHub repository]({cfg['app']['github_url']})")

    st.header("Dataset & metrics")
    ds = meta["dataset"]
    st.write(f"Source: `{ds['source']}`")
    st.dataframe(pd.DataFrame(ds["counts"]).T, width="stretch")
    st.write(f"Writer-separated split: {ds['train_writers']} train writers, {ds['test_writers']} test writers")
    m = meta["models"].get(f"{feature_set}_{model_key}")
    if m:
        tm = m["test_metrics"]
        st.write("Held-out writers (this model):")
        st.dataframe(pd.DataFrame({"metric": ["macro-F1", "accuracy", "ROC-AUC"],
                                   "value": [tm["f1_macro"], tm["accuracy"], tm["roc_auc"]]}).round(3),
                     hide_index=True, width="stretch")
    st.caption(f"Trained {meta['trained_at']}")

pipe = load_model(f"{feature_set}_{model_key}")

# ---- input ---------------------------------------------------------------------
if "text" not in st.session_state:
    st.session_state.text = ""
cols = st.columns(len(EXAMPLES))
for col, (name, txt) in zip(cols, EXAMPLES.items()):
    if col.button(name, width="stretch"):
        st.session_state.text = txt
text = st.text_area("Paste an English passage (ideally 120+ words)", key="text", height=240)
v = validate_text(text, cfg["app"]["min_words"], cfg["app"]["warn_words"])
st.caption(f"Word count: {v.words}")
analyze = st.button("Analyze writing", type="primary")

if analyze:
    if not v.ok:
        st.error(v.message)
        st.stop()
    if v.level == "warning":
        st.warning(v.message)

    proba = pipe.predict_proba(pd.Series([text]))[0]
    classes = list(pipe.classes_)
    conf = {c: float(p) for c, p in zip(classes, proba)}
    pred = max(conf, key=conf.get)
    pretty = {"native": "Native", "non_native": "Non-native"}

    left, right = st.columns(2)
    with left:
        st.subheader("Model confidence")
        st.caption("A confidence score from patterns in the training data, not a certainty.")
        for c in ("native", "non_native"):
            st.write(f"{pretty[c]}: **{conf.get(c, 0):.0%}**")
            st.progress(conf.get(c, 0.0))
        st.info(f"The model leans toward **{pretty[pred]}** for this passage.")
    with right:
        st.subheader("Style analysis")
        feats = style_extractor().transform([text]).iloc[0]
        st.dataframe(pd.DataFrame(style_summary(feats), columns=["statistic", "value"]).round(2),
                     hide_index=True, width="stretch")

    st.subheader("Feature-group importance")
    abl_path = results_dir / "ablation.csv"
    g1, g2 = st.columns(2)
    with g1:
        gi = coef_group_importance(pipe)
        if not gi.empty:
            st.caption("Sum of |coefficient| per group in this model")
            st.bar_chart(gi.set_index("group")["abs_coef_sum"])
    with g2:
        if abl_path.exists():
            abl = pd.read_csv(abl_path)
            st.caption("Drop in macro-F1 when a handcrafted group is removed (experiment E5)")
            st.bar_chart(abl.set_index("group")["f1_drop_mean"])

    st.subheader("Top contributing features")
    st.caption("Features associated with the model's prediction on this dataset, not causes. "
               "Contribution = coefficient x feature value (standardised for handcrafted features, "
               "TF-IDF weight for character n-grams).")
    contrib = local_contributions(pipe, text, k=8)
    if contrib.empty:
        st.write("No non-zero contributions for this text.")
    else:
        for c in ("native", "non_native"):
            part = contrib[contrib.toward == c][["feature", "group", "value", "contribution"]]
            st.markdown(f"**Pushing toward {pretty[c]}**")
            st.dataframe(part.round(4), hide_index=True, width="stretch")

with st.expander("About this demo"):
    st.markdown(
        "**Method.** Texts are described by five groups of handcrafted stylometric features and by "
        "character 3-5-gram TF-IDF. Logistic Regression and a calibrated Linear SVM are trained on a "
        "writer-balanced dataset and evaluated on writers never seen in training. See `docs/methodology.md`."
    )
    res_md = results_dir / "results.md"
    if res_md.exists():
        st.markdown(res_md.read_text(encoding="utf-8"))
    else:
        st.write("Results table not generated yet (run `python tasks.py evaluate`).")
    st.markdown(
        "**Limitations.** Results reflect one corpus, one genre (short argumentative essays) and its "
        "collection conditions. Differences can come from proficiency, topic, education, or editing "
        "rather than native status. Short texts are unreliable. See `docs/ethics_and_limitations.md`."
    )
