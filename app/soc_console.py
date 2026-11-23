"""
SOC Live Console — ML-based Intrusion Detection (v5)
====================================================
Replays the sealed UNSW-NB15 test partition as if it were live traffic: flows
stream in, the production XGBoost pipeline scores each one, alerts fire above the
decision threshold, and the analyst watches the operational trade-off happen.

Corrected for the v5 bundle:
  * bundle key is "pipeline" (end-to-end), not "model" + "preprocessor"
  * threshold read from the bundle (0.630), not the abandoned 0.83
  * apply_prep() replicated from the notebook so raw CSV columns match the pipeline
  * log1p_clip registered into __main__ before joblib.load, or unpickling fails
  * SHAP computed on the transformed matrix by splitting the pipeline
  * the sttl message now reflects the ablation result instead of contradicting it

Run locally (from the repository root):
    streamlit run app/soc_console.py

Paths default to models/ and data/ and can be overridden with the
IDS_BUNDLE_PATH and IDS_TEST_CSV environment variables (e.g. on Colab:
/content/drive/MyDrive/data/...).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "BUNDLE_PATH": os.environ.get("IDS_BUNDLE_PATH", "models/ids_xgb_bundle.joblib"),
    "TEST_CSV_PATH": os.environ.get("IDS_TEST_CSV", "data/UNSW_NB15_testing-set.csv"),
    "LABEL_COL": "label",
    "ATTACK_CAT_COL": "attack_cat",
    "ID_COL": "id",
    "FALLBACK_THRESHOLD": 0.630,
    "REPLAY_SEED": 42,
}

COLORS = {
    "bg": "#0d1420", "panel": "#151d2b", "ink": "#e6ecf5", "muted": "#7d8aa0",
    "grid": "#243247", "caught": "#e5484d", "false": "#f5a623",
    "clean": "#3bb0c9", "missed": "#a06bd6",
}

CSS = f"""
<style>
  .stApp {{ background:{COLORS['bg']}; }}
  html, body, [class*="css"] {{ color:{COLORS['ink']}; }}
  h1,h2,h3,h4 {{ color:{COLORS['ink']}; letter-spacing:.2px; }}
  .kpi {{ background:{COLORS['panel']}; border:1px solid {COLORS['grid']};
          border-radius:12px; padding:14px 16px; }}
  .kpi .label {{ color:{COLORS['muted']}; font-size:.72rem;
                 text-transform:uppercase; letter-spacing:.12em; }}
  .kpi .value {{ font-size:1.7rem; font-weight:700; font-variant-numeric:tabular-nums;
                 font-family:ui-monospace,Menlo,monospace; }}
  .alert-card {{ background:{COLORS['panel']}; border-left:4px solid {COLORS['grid']};
                 border-radius:8px; padding:8px 12px; margin-bottom:7px;
                 font-family:ui-monospace,Menlo,monospace; font-size:.8rem; }}
  .tag {{ display:inline-block; padding:1px 8px; border-radius:999px;
          font-size:.68rem; font-weight:700; }}
  /* Streamlit's default caption and body colours are near-black and vanish on
     this canvas - unreadable on a projector. Force them to the muted/ink ramp. */
  [data-testid="stCaptionContainer"],
  [data-testid="stCaptionContainer"] p,
  small, .stCaption, .stCaption p {{ color:{COLORS['muted']} !important; }}
  .stMarkdown p, .stMarkdown li, .stMarkdown span,
  [data-testid="stMarkdownContainer"] p {{ color:{COLORS['ink']}; }}
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label {{ color:{COLORS['muted']} !important; }}
  [data-testid="stMetricValue"] {{ color:{COLORS['ink']}; }}
</style>
"""


# ---------------------------------------------------------------------------
# CRITICAL: the pipeline contains a FunctionTransformer wrapping log1p_clip.
# joblib looks it up in __main__ at load time. Without this, loading raises
# AttributeError: Can't get attribute 'log1p_clip' on module '__main__'.
# ---------------------------------------------------------------------------
def dark_chart(chart, height=280):
    """Charts default to a white plot panel, which reads as a bright rectangle on
    this canvas and is unusable on a projector. Theme them to the app palette and
    pass theme=None at render time so Streamlit does not override it."""
    return (chart
            .properties(height=height, background=COLORS["bg"])
            .configure_view(strokeWidth=0, fill=COLORS["bg"])
            .configure_axis(labelColor=COLORS["ink"], titleColor=COLORS["muted"],
                            gridColor=COLORS["grid"], domainColor=COLORS["grid"],
                            tickColor=COLORS["grid"], labelFontSize=12,
                            titleFontSize=12)
            .configure_legend(labelColor=COLORS["ink"], titleColor=COLORS["muted"],
                              labelFontSize=12))


def log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))


sys.modules["__main__"].log1p_clip = log1p_clip


# ---------------------------------------------------------------------------
# Preprocessing — must mirror apply_prep() in the notebook exactly
# ---------------------------------------------------------------------------
def apply_prep(df: pd.DataFrame, params: dict, drop_redundant: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=[CONFIG["ID_COL"]], inplace=True, errors="ignore")
    df["service"] = df["service"].replace("-", "none")
    df["proto"] = np.where(df["proto"].isin(params["keep_proto"]), df["proto"], "other")
    if drop_redundant:
        df.drop(columns=params["drop_redundant"], inplace=True, errors="ignore")
    return df


def load_artifacts(config: dict) -> dict:
    import joblib

    bundle = joblib.load(config["BUNDLE_PATH"])
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            "Bundle does not contain a 'pipeline' key. This console expects the v5 "
            "bundle written by §17 of the notebook."
        )

    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", config["FALLBACK_THRESHOLD"]))
    prep_params = bundle["prep_params"]

    raw = pd.read_csv(config["TEST_CSV_PATH"])
    # The sealed test partition is stored grouped by attack category, so replaying
    # it in file order makes any prefix unrepresentative of the partition as a whole.
    # A fixed-seed permutation restores the 55/45 attack-normal mix in every window.
    # This changes arrival order only; the evaluation in the notebook is unaffected.
    raw = (raw.sample(frac=1.0, random_state=config["REPLAY_SEED"])
              .reset_index(drop=True))
    y = raw[config["LABEL_COL"]].astype(int).to_numpy()
    fam = raw[config["ATTACK_CAT_COL"]].fillna("Normal").astype(str).to_numpy()
    flow_id = (raw[config["ID_COL"]].to_numpy()
               if config["ID_COL"] in raw.columns else np.arange(len(raw)))

    clean = apply_prep(raw, prep_params)
    X = clean.drop(columns=[c for c in (config["LABEL_COL"], config["ATTACK_CAT_COL"])
                            if c in clean.columns])

    return {
        "pipeline": pipeline, "threshold": threshold, "bundle": bundle,
        "X": X, "X_raw": raw, "y": y, "attack_cat": fam, "flow_id": flow_id,
    }


def compute_probabilities(art: dict) -> np.ndarray:
    return np.asarray(art["pipeline"].predict_proba(art["X"])[:, 1], dtype=float)


def live_metrics(y_true, proba, threshold, upto) -> dict:
    if upto <= 0:
        return dict(processed=0, alerts=0, tp=0, fp=0, fn=0, tn=0,
                    detection=0.0, fpr=0.0, precision=0.0)
    yt = y_true[:upto]
    pred = (proba[:upto] >= threshold).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    attacks, normals, alerts = tp + fn, fp + tn, tp + fp
    return dict(processed=upto, alerts=alerts, tp=tp, fp=fp, fn=fn, tn=tn,
                detection=(tp / attacks) if attacks else 0.0,
                fpr=(fp / normals) if normals else 0.0,
                precision=(tp / alerts) if alerts else 0.0)


def shap_top_features(art: dict, row_idx: int, k: int = 8):
    """Split the pipeline: transform with prep, explain the model. Never raises."""
    try:
        import shap
        import scipy.sparse as sp

        pipe = art["pipeline"]
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        names = [f.split("__")[-1] for f in prep.get_feature_names_out()]

        row = art["X"].iloc[[row_idx]]
        Xt = prep.transform(row)
        Xt = Xt.toarray() if sp.issparse(Xt) else np.asarray(Xt)

        vals = shap.TreeExplainer(model).shap_values(Xt)
        vals = np.asarray(vals[1] if isinstance(vals, list) else vals).reshape(-1)
        if len(names) != len(vals):
            names = [f"f{i}" for i in range(len(vals))]
        order = np.argsort(np.abs(vals))[::-1][:k]
        raw_vals = {n: (row.iloc[0][n] if n in row.columns else None) for n in
                    [names[i] for i in order]}
        return [(names[i], float(vals[i]), raw_vals.get(names[i])) for i in order], None
    except Exception as e:
        return [], f"SHAP unavailable ({type(e).__name__})"


def _kpi(col, label, value, color=None):
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color or COLORS["ink"]}">{value}</div></div>',
        unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="SOC Live Console — ML IDS", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        art = st.session_state.get("art")
        if art is None:
            with st.spinner("Loading v5 bundle and sealed test partition…"):
                art = load_artifacts(CONFIG)
                art["proba"] = compute_probabilities(art)
                st.session_state["art"] = art
    except Exception as e:
        st.error(f"Could not load artifacts. Check CONFIG paths.\n\nDetails: {e}")
        st.stop()

    proba, y, fam = art["proba"], art["y"], art["attack_cat"]
    n = len(proba)
    b = art["bundle"]

    ss = st.session_state
    ss.setdefault("cursor", 0)
    ss.setdefault("running", False)
    ss.setdefault("threshold", art["threshold"])

    st.markdown("## SOC Live Console — ML-based Intrusion Detection")
    st.caption(
        f"Sealed UNSW-NB15 test partition ({n:,} flows) replayed as live traffic · "
        f"scored by XGBoost bundle {b.get('version','v5')} · "
        f"operating threshold {art['threshold']:.3f} "
        f"({b.get('threshold_provenance', {}).get('method', 'FPR-constrained')}) · "
        "ground truth used only to grade the simulation."
    )

    with st.sidebar:
        st.markdown("### Console controls")
        ss.threshold = st.slider("Decision threshold", 0.05, 0.99,
                                 float(ss.threshold), 0.01,
                                 help="The SOC trade-off dial. The bundle's chosen "
                                      "value is 0.630, selected on validation under "
                                      "a 5% false-positive budget.")
        if abs(ss.threshold - art["threshold"]) > 1e-9:
            st.info(f"Moved off the calibrated threshold "
                    f"({art['threshold']:.3f}). Demonstration only — the reported "
                    f"results use the calibrated value.")
        speed = st.select_slider("Stream speed",
                                 options=["slow", "normal", "fast", "turbo"],
                                 value="normal")
        batch = {"slow": 20, "normal": 60, "fast": 150, "turbo": 400}[speed]
        interval = {"slow": 0.9, "normal": 0.5, "fast": 0.3, "turbo": 0.12}[speed]

        c1, c2 = st.columns(2)
        if c1.button("Play" if not ss.running else "Pause", use_container_width=True):
            ss.running = not ss.running
        if c2.button("Step", use_container_width=True):
            ss.cursor = min(n, ss.cursor + batch); ss.running = False
        if st.button("Jump to next false alarm", use_container_width=True):
            # Advance the cursor to just past the next false positive so that it
            # lands at the head of the alert feed and in the inspection dropdown.
            tail = np.flatnonzero((y[ss.cursor:] == 0) &
                                  (proba[ss.cursor:] >= ss.threshold))
            if tail.size:
                ss.cursor = min(n, ss.cursor + int(tail[0]) + 1)
                ss.running = False
            else:
                st.warning("No further false alarm in the remaining stream.")
        if st.button("Reset stream", use_container_width=True):
            ss.cursor = 0; ss.running = False

        st.markdown("---")
        st.markdown(
            f'<span class="tag" style="background:{COLORS["caught"]}22;color:{COLORS["caught"]}">attack caught</span> '
            f'<span class="tag" style="background:{COLORS["false"]}22;color:{COLORS["false"]}">false alarm</span><br>'
            f'<span class="tag" style="background:{COLORS["missed"]}22;color:{COLORS["missed"]}">missed</span> '
            f'<span class="tag" style="background:{COLORS["clean"]}22;color:{COLORS["clean"]}">clean</span>',
            unsafe_allow_html=True)

    upto = ss.cursor
    m = live_metrics(y, proba, ss.threshold, upto)

    k = st.columns(6)
    _kpi(k[0], "Flows processed", f"{m['processed']:,}")
    _kpi(k[1], "Alerts raised", f"{m['alerts']:,}")
    _kpi(k[2], "Attacks caught", f"{m['tp']:,}", COLORS["caught"])
    _kpi(k[3], "False alarms", f"{m['fp']:,}", COLORS["false"])
    _kpi(k[4], "Detection rate", f"{m['detection']*100:.1f}%", COLORS["caught"])
    _kpi(k[5], "False-positive rate", f"{m['fpr']*100:.2f}%", COLORS["false"])

    st.markdown(
        f"<div style='height:6px;background:{COLORS['grid']};border-radius:6px;margin:10px 0'>"
        f"<div style='height:6px;width:{(upto/n*100) if n else 0:.1f}%;"
        f"background:{COLORS['clean']};border-radius:6px'></div></div>",
        unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Live alert feed")
        window = range(max(0, upto - 400), upto)
        recent = [i for i in window if proba[i] >= ss.threshold][-14:][::-1]
        if not recent:
            st.markdown(f"<div class='alert-card' style='color:{COLORS['muted']}'>"
                        f"No alerts in the current window.</div>",
                        unsafe_allow_html=True)
        for i in recent:
            true_attack = y[i] == 1
            color = COLORS["caught"] if true_attack else COLORS["false"]
            label = fam[i] if true_attack else "FALSE ALARM"
            st.markdown(
                f"<div class='alert-card' style='border-left-color:{color}'>"
                f"<span style='color:{color};font-weight:700'>{label}</span> "
                f"<span style='color:{COLORS['muted']}'>· flow {art['flow_id'][i]} "
                f"· score {proba[i]:.3f} · threshold {ss.threshold:.2f}</span></div>",
                unsafe_allow_html=True)

        st.markdown("#### Why was this flow flagged?")
        if recent:
            pick = st.selectbox(
                "Inspect an alert", recent,
                format_func=lambda i: f"flow {art['flow_id'][i]} · "
                                      f"{fam[i] if y[i]==1 else 'false alarm'} · "
                                      f"score {proba[i]:.3f}")
            feats, note = shap_top_features(art, pick, k=8)
            if feats:
                import altair as alt
                df = pd.DataFrame([(f, v) for f, v, _ in feats],
                                  columns=["feature", "contribution"])
                # Ordered by |contribution| and drawn horizontally: st.bar_chart would
                # re-sort the index alphabetically and destroy the ranking.
                chart = (alt.Chart(df).mark_bar()
                         .encode(
                             y=alt.Y("feature:N", sort="-x", title=None),
                             x=alt.X("contribution:Q",
                                     title="SHAP contribution (log-odds)"),
                             tooltip=["feature", "contribution"],
                             color=alt.condition(alt.datum.contribution > 0,
                                                 alt.value(COLORS["caught"]),
                                                 alt.value(COLORS["clean"])))
                         )
                st.altair_chart(dark_chart(chart, 280), use_container_width=True,
                                theme=None)
                st.caption("Red pushes the score towards attack, blue towards normal. "
                           "SHAP contributions are in log-odds, not probability "
                           "points, and are not causal.")
                names = [str(f).lower() for f, _, _ in feats]
                if "sttl" in names[0]:
                    lead = ("`sttl` is the largest contributor to this decision, as it "
                            "is the strongest feature of the model globally.")
                elif any("sttl" in n for n in names):
                    lead = ("`sttl` contributes to this decision without leading it; "
                            "the largest contributor here is "
                            f"`{feats[0][0]}`.")
                else:
                    lead = None
                if lead:
                    st.info(
                        lead + " Either way the model does not depend on `sttl`: the "
                        "ablation study (§13) shows that removing it and retraining "
                        "costs 0.0001 ROC-AUC, because correlated features substitute "
                        "for it.",
                        icon="ℹ️")
            elif note:
                st.caption(note)

    with right:
        st.markdown("#### Attack families — caught vs missed")
        if upto > 0:
            pred = (proba[:upto] >= ss.threshold).astype(int)
            yt, ft = y[:upto], fam[:upto]
            rows = []
            for f in sorted(set(ft[yt == 1])):
                mask = (ft == f) & (yt == 1)
                total = int(mask.sum())
                caught = int((mask & (pred == 1)).sum())
                if total:
                    rows.append({"family": f, "caught": caught, "missed": total - caught})
            if rows:
                import altair as alt
                long = (pd.DataFrame(rows)
                          .melt(id_vars="family", value_vars=["caught", "missed"],
                                var_name="outcome", value_name="flows"))
                fam_chart = (alt.Chart(long).mark_bar()
                             .encode(
                                 x=alt.X("family:N", sort="-y", title=None,
                                         axis=alt.Axis(labelAngle=-40)),
                                 y=alt.Y("flows:Q", title="flows"),
                                 color=alt.Color("outcome:N", title=None,
                                     scale=alt.Scale(domain=["caught", "missed"],
                                         range=[COLORS["caught"], COLORS["missed"]])),
                                 tooltip=["family", "outcome", "flows"]))
                st.altair_chart(dark_chart(fam_chart, 260), use_container_width=True,
                                theme=None)
                weak = min(rows, key=lambda r: r["caught"] / max(r["caught"] + r["missed"], 1))
                st.caption(f"Weakest coverage: **{weak['family']}** "
                           f"({weak['caught']}/{weak['caught']+weak['missed']} caught).")

        st.markdown("#### Analyst alert load")
        load = (m["fp"] / m["alerts"] * 100) if m["alerts"] else 0.0
        bar = COLORS["clean"] if load < 20 else (COLORS["false"] if load < 50 else COLORS["caught"])
        st.markdown(
            f"<div style='background:{COLORS['panel']};border:1px solid {COLORS['grid']};"
            f"border-radius:10px;padding:12px 14px'>"
            f"<div style='color:{COLORS['muted']};font-size:.72rem'>SHARE OF ALERTS THAT ARE FALSE</div>"
            f"<div style='font-family:ui-monospace;font-size:1.6rem;font-weight:700;color:{bar}'>{load:.1f}%</div>"
            f"<div style='height:8px;background:{COLORS['grid']};border-radius:6px;margin-top:8px'>"
            f"<div style='height:8px;width:{min(load,100):.0f}%;background:{bar};border-radius:6px'></div>"
            f"</div></div>", unsafe_allow_html=True)
        st.caption("On this benchmark 55% of flows are attacks. At a realistic base "
                   "rate of 1 in 1000, the same model would yield roughly 0.5% "
                   "precision — around 200 false alarms per real attack.")

    if ss.running and ss.cursor < n:
        ss.cursor = min(n, ss.cursor + batch)
        time.sleep(interval)
        st.rerun()
    elif ss.cursor >= n and n > 0:
        ss.running = False
        st.success("End of stream — full sealed test partition replayed.")


if __name__ == "__main__":
    main()
