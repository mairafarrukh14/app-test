"""QuantMind — Streamlit Dashboard
===================================
Drop this file into:  Uni Project/quantmind_prototype/streamlit_app.py
Run with:             streamlit run streamlit_app.py

The app runs the full pipeline in-browser:
  Load / refresh data  →  Train PPO agent  →  Backtest  →  XAI (Shapley)
All stages are cached so re-opening the app is instant.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── make sure `src/` is on the path regardless of where streamlit is run from ──
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import config, data, train, backtest, explain, metrics  # noqa: E402

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantMind — RL Portfolio Advisor",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Minimal inline CSS (keeps it clean without external deps)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Metric cards */
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
/* Section headers */
h2 { border-bottom: 2px solid #3b6ea5; padding-bottom: 4px; }
/* Sidebar title */
.sidebar-title { font-size: 1.1rem; font-weight: 600; color: #3b6ea5; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Sidebar — controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ QuantMind Settings</p>', unsafe_allow_html=True)

    timesteps = st.select_slider(
        "PPO training steps",
        options=[5_000, 25_000, 50_000, 100_000, 150_000, 200_000],
        value=50_000,
        help="More steps = better agent, longer wait. 5k is a fast smoke-test.",
    )

    run_shap = st.checkbox("Run XAI (Shapley)", value=True,
                           help="Uncheck to skip the slower attribution stage.")

    refresh_data = st.button("🔄 Re-download price data",
                             help="Force-refresh from Yahoo Finance (needs internet).")

    st.divider()
    st.caption("Assets: " + " · ".join(config.TICKERS))
    st.caption(f"Train → {config.TRAIN_TEST_SPLIT}")
    st.caption(f"Seed: {config.SEED}")
    st.caption(f"Max weight: {config.MAX_WEIGHT:.0%} | TC: {config.TRANSACTION_COST:.1%}")

# ─────────────────────────────────────────────
#  Cached pipeline stages
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_prices(force: bool):
    return data.load_prices(force_refresh=force)


@st.cache_data(show_spinner=False)
def cached_build_features(_prices):
    return data.build_features(_prices)


@st.cache_resource(show_spinner=False)
def cached_train(tr_feat_hash: int, tr_rets_hash: int, n_steps: int):
    """Cache is keyed on data hashes + step count so re-runs only when needed."""
    # Retrieve from session_state (hashing trick: we store data there)
    tr_feat = st.session_state["_tr_feat"]
    tr_rets = st.session_state["_tr_rets"]
    return train.train_agent(tr_feat, tr_rets, total_timesteps=n_steps, verbose=0)


@st.cache_data(show_spinner=False)
def cached_backtest(_model, te_feat, te_rets):
    agent_rets, weights = backtest.run_agent(_model, te_feat, te_rets)
    bh = backtest.buy_and_hold(te_rets).reindex(agent_rets.index).fillna(0.0)
    best = backtest.best_single_asset(te_rets).reindex(agent_rets.index).fillna(0.0)
    strategies = {agent_rets.name: agent_rets, bh.name: bh, best.name: best}
    table = metrics.summary_table(strategies, rf_daily=config.RISK_FREE_DAILY)
    avg_turnover = float(weights.diff().abs().sum(axis=1).mean())
    return agent_rets, weights, bh, best, table, avg_turnover


@st.cache_data(show_spinner=False)
def cached_explain(_model, tr_feat, tr_rets_vals, te_feat, te_rets_vals,
                   te_rets_cols, n_quick: bool):
    # Reconstruct DataFrames (only serialisable types can be cache keys)
    tr_rets = pd.DataFrame(tr_rets_vals[0], index=tr_rets_vals[1], columns=tr_rets_vals[2])
    te_rets = pd.DataFrame(te_rets_vals[0], index=te_rets_vals[1], columns=te_rets_cols)
    phi_all, X_explain, sel, feat_names, tickers = explain.explain(
        _model, tr_feat, tr_rets, te_feat, te_rets,
        n_background=15 if n_quick else 30,
        n_explain=10 if n_quick else 40,
        n_perm=24 if n_quick else 64,
    )
    importance = np.abs(phi_all).mean(axis=(0, 2))
    last_phi = phi_all[-1]
    last_obs = X_explain[-1]
    rationale = explain.narrate_decision(_model, last_obs, tickers, last_phi, feat_names)
    return phi_all, feat_names, tickers, importance, last_phi, last_obs, rationale


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.title("📈 QuantMind — RL Portfolio Advisor")
st.caption("PPO reinforcement-learning agent with Shapley-value explainability | CM3070 prototype")

# ─────────────────────────────────────────────
#  Stage 1 — Data
# ─────────────────────────────────────────────
st.header("1 · Data")

with st.spinner("Loading price data …"):
    t0 = time.time()
    prices, source = cached_load_prices(refresh_data)
    features, log_rets = cached_build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]

# Store in session_state for the training cache (avoids pickling large arrays)
st.session_state["_tr_feat"] = tr_feat
st.session_state["_tr_rets"] = tr_rets

src_colour = {"yfinance": "green", "cache": "blue", "simulated": "orange"}.get(source, "gray")
st.markdown(f"Data source: :{src_colour}[**{source}**] &nbsp;|&nbsp; "
            f"Universe: **{', '.join(config.TICKERS)}** &nbsp;|&nbsp; "
            f"{prices.index.min().date()} → {prices.index.max().date()}")

col1, col2, col3 = st.columns(3)
col1.metric("Training days", f"{tr_feat.shape[0]:,}")
col2.metric("Test days", f"{te_feat.shape[0]:,}")
col3.metric("Features / asset", tr_feat.shape[2])

with st.expander("Raw price chart"):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for tkr in prices.columns:
        (prices[tkr] / prices[tkr].iloc[0]).plot(ax=ax, label=tkr, linewidth=1.2)
    ax.axvline(pd.Timestamp(config.TRAIN_TEST_SPLIT), color="red",
               linestyle="--", linewidth=1, label="Train / Test split")
    ax.set_title("Normalised price history (rebased to 1)")
    ax.set_ylabel("Relative price")
    ax.legend(ncol=len(prices.columns) + 1, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────
#  Stage 2 — Train
# ─────────────────────────────────────────────
st.header("2 · PPO Training")

tr_hash = int(hash(tr_feat.tobytes()[:1024]))
tr_rets_hash = int(hash(tr_rets.values.tobytes()[:1024]))

with st.spinner(f"Training PPO for {timesteps:,} steps … (this may take a minute)"):
    t1 = time.time()
    model = cached_train(tr_hash, tr_rets_hash, timesteps)
    train_time = time.time() - t1

st.success(f"✅ Agent trained in **{train_time:.1f}s** · {timesteps:,} timesteps · "
           f"Policy: MLP [128 × 128] · Seed {config.SEED}")

# ─────────────────────────────────────────────
#  Stage 3 — Backtest
# ─────────────────────────────────────────────
st.header("3 · Backtest Results")

with st.spinner("Running backtest …"):
    agent_rets, weights, bh, best, table, avg_turnover = cached_backtest(
        model, te_feat, te_rets)

# ── KPI cards ──────────────────────────────────────────────────────────────
ppo_row = table.loc["QuantMind PPO"]
bh_row = table.loc["Buy & Hold (1/N)"]

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Return",
          f"{ppo_row['Total Return']:.1%}",
          delta=f"{ppo_row['Total Return'] - bh_row['Total Return']:+.1%} vs B&H")
k2.metric("CAGR",       f"{ppo_row['CAGR']:.1%}")
k3.metric("Sharpe",     f"{ppo_row['Sharpe']:.2f}",
          delta=f"{ppo_row['Sharpe'] - bh_row['Sharpe']:+.2f} vs B&H")
k4.metric("Sortino",    f"{ppo_row['Sortino']:.2f}")
k5.metric("Max Drawdown", f"{ppo_row['Max Drawdown']:.1%}")
k6.metric("Ann. Vol",   f"{ppo_row['Ann. Volatility']:.1%}")

# ── Equity curve ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
init = config.INITIAL_CASH
ax.plot(agent_rets.index,
        metrics.equity_curve(agent_rets.values, init),
        label="QuantMind PPO", linewidth=2.4, color="#3b6ea5")
ax.plot(bh.index,
        metrics.equity_curve(bh.values, init),
        label="Buy & Hold (1/N)", linewidth=1.4, color="#e07b39", linestyle="--")
ax.plot(best.index,
        metrics.equity_curve(best.values, init),
        label=best.name, linewidth=1.2, color="#6abf6a", linestyle=":")
ax.set_title("Portfolio value on held-out test period")
ax.set_ylabel("Portfolio value (£)")
ax.set_xlabel("Date")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Full metrics table ─────────────────────────────────────────────────────
with st.expander("Full metrics table"):
    fmt = {c: "{:.4f}" for c in table.columns}
    fmt["Total Return"] = "{:.2%}"
    fmt["CAGR"] = "{:.2%}"
    fmt["Ann. Volatility"] = "{:.2%}"
    fmt["Max Drawdown"] = "{:.2%}"
    st.dataframe(
        table.style.format({
            "Total Return": "{:.2%}", "CAGR": "{:.2%}",
            "Ann. Volatility": "{:.2%}", "Max Drawdown": "{:.2%}",
            "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
        }).background_gradient(cmap="RdYlGn", subset=["Total Return", "Sharpe", "Max Drawdown"]),
        use_container_width=True,
    )
    st.caption(f"Avg daily turnover: **{avg_turnover:.4f}** "
               f"({avg_turnover * 100:.2f}% of portfolio rebalanced per day)")

# ── Allocation over time ───────────────────────────────────────────────────
st.subheader("Portfolio allocation over time")

fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.stackplot(weights.index, weights.T.values,
              labels=weights.columns, alpha=0.85)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Weight")
ax2.legend(loc="upper center", ncol=len(weights.columns), fontsize=8,
           bbox_to_anchor=(0.5, 1.12))
ax2.grid(alpha=0.2)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ── Current allocation doughnut ────────────────────────────────────────────
st.subheader("Current allocation (most recent day)")
curr = weights.iloc[-1]
fig3, ax3 = plt.subplots(figsize=(5, 5))
wedge_colours = ["#3b6ea5", "#e07b39", "#6abf6a", "#c94040", "#9b59b6", "#888"]
ax3.pie(curr.values,
        labels=[f"{c}\n{v:.1%}" for c, v in zip(curr.index, curr.values)],
        colors=wedge_colours[:len(curr)],
        startangle=90,
        wedgeprops=dict(width=0.55))
ax3.set_title("Current portfolio weights")
col_pie, col_curr = st.columns([1, 1])
with col_pie:
    st.pyplot(fig3)
    plt.close(fig3)
with col_curr:
    curr_df = curr.reset_index()
    curr_df.columns = ["Asset", "Weight"]
    curr_df["Weight %"] = (curr_df["Weight"] * 100).map("{:.1f}%".format)
    curr_df["Action"] = curr_df["Weight"].apply(
        lambda w: "🔼 Overweight" if w >= 0.30 else ("🟡 Hold" if w >= 0.12 else "🔽 Underweight")
    )
    st.dataframe(curr_df[["Asset", "Weight %", "Action"]],
                 use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
#  Stage 4 — XAI
# ─────────────────────────────────────────────
st.header("4 · Explainability (XAI)")

if not run_shap:
    st.info("XAI is disabled — enable 'Run XAI (Shapley)' in the sidebar and rerun.")
else:
    quick = timesteps <= 5_000  # use tiny XAI budget for smoke-test runs

    with st.spinner("Computing Shapley-value attributions … (30 – 90 s for full run)"):
        phi_all, feat_names, tickers, importance, last_phi, last_obs, rationale = cached_explain(
            model,
            tr_feat,
            (tr_rets.values, tr_rets.index, list(tr_rets.columns)),
            te_feat,
            (te_rets.values, te_rets.index, list(te_rets.columns)),
            list(te_rets.columns),
            quick,
        )

    # ── Global feature importance bar chart ────────────────────────────────
    st.subheader("Global feature importance")
    st.caption("Mean |Shapley value| over sampled test decisions — "
               "larger = bigger average impact on the allocation.")

    top_k = 15
    order = np.argsort(importance)[::-1][:top_k]
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(
        [feat_names[i] for i in order][::-1],
        importance[order][::-1],
        color="#3b6ea5",
    )
    ax_imp.set_xlabel("Mean |Shapley value|")
    ax_imp.set_title("QuantMind XAI — feature importance")
    ax_imp.grid(axis="x", alpha=0.25)
    fig_imp.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    # ── Per-asset Shapley breakdown (last day) ──────────────────────────────
    st.subheader("Per-asset attribution — most recent decision")

    f_fn = explain.make_policy_fn(model)
    last_weights = f_fn(last_obs.reshape(1, -1))[0]
    labels = tickers + ["CASH"]

    n_assets = len(tickers)
    fig_att, axes = plt.subplots(1, n_assets, figsize=(3.5 * n_assets, 4), sharey=False)
    for i, (tkr, ax_a) in enumerate(zip(tickers, axes)):
        contribs = last_phi[:, i]
        own_idx = [k for k, n in enumerate(feat_names) if n.startswith(f"{tkr}:")]
        top_own = sorted(own_idx, key=lambda k: abs(contribs[k]), reverse=True)[:5]
        vals = [contribs[k] for k in top_own]
        names = [feat_names[k].split(":")[1] for k in top_own]
        colours = ["#3b6ea5" if v > 0 else "#c94040" for v in vals]
        ax_a.barh(names[::-1], vals[::-1], color=colours[::-1])
        ax_a.axvline(0, color="black", linewidth=0.6)
        ax_a.set_title(f"{tkr} ({last_weights[i]:.0%})", fontsize=10)
        ax_a.set_xlabel("φ (Shapley)")
        ax_a.grid(axis="x", alpha=0.2)
    fig_att.suptitle("Feature contributions to each asset's weight (last day)", y=1.02)
    fig_att.tight_layout()
    st.pyplot(fig_att)
    plt.close(fig_att)

    # ── Natural-language rationale ──────────────────────────────────────────
    st.subheader("Natural-language rationale (most recent decision)")
    st.info(rationale)

    # ── Shapley heatmap across all explained days ───────────────────────────
    with st.expander("Shapley heatmap — all sampled test decisions"):
        st.caption("Rows = sampled test days · Columns = features · Colour = mean |φ| over assets")
        heat = np.abs(phi_all).mean(axis=2)   # (n_explain, M)
        # show only top-20 features by importance
        top20 = np.argsort(importance)[::-1][:20]
        heat_df = pd.DataFrame(
            heat[:, top20],
            columns=[feat_names[i] for i in top20],
        )
        fig_h, ax_h = plt.subplots(figsize=(12, max(4, len(heat_df) * 0.25 + 1)))
        im = ax_h.imshow(heat_df.values, aspect="auto", cmap="YlOrRd")
        ax_h.set_yticks([])
        ax_h.set_xticks(range(len(heat_df.columns)))
        ax_h.set_xticklabels(heat_df.columns, rotation=45, ha="right", fontsize=7)
        plt.colorbar(im, ax=ax_h, label="|φ|")
        ax_h.set_title("Shapley magnitude heatmap")
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

# ─────────────────────────────────────────────
#  Stage 5 — Run summary JSON
# ─────────────────────────────────────────────
st.header("5 · Run Summary")

summary = {
    "data_source": source,
    "train_days": int(tr_feat.shape[0]),
    "test_days": int(te_feat.shape[0]),
    "timesteps": int(timesteps),
    "metrics": json.loads(table.to_json(orient="index")),
    "avg_daily_turnover": round(avg_turnover, 6),
    "total_runtime_sec": round(time.time() - t0, 1),
}

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.download_button(
        "⬇️ Download run_summary.json",
        data=json.dumps(summary, indent=2),
        file_name="run_summary.json",
        mime="application/json",
    )

with col_dl2:
    metrics_csv = table.to_csv()
    st.download_button(
        "⬇️ Download metrics.csv",
        data=metrics_csv,
        file_name="metrics.csv",
        mime="text/csv",
    )

with st.expander("View run summary JSON"):
    st.json(summary)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "QuantMind · CM3070 Final Project · "
    "PPO via Stable-Baselines3 · "
    "Shapley XAI implemented from scratch in NumPy · "
    f"Total runtime: {time.time() - t0:.1f}s"
)
