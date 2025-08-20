import streamlit as st
import pandas as pd
import numpy as np
from math import erf
from typing import List, Dict, Tuple

# ============================
# THEME / STYLES (Dark Neon)
# ============================
st.set_page_config(page_title="Draft Assistant Live — UX", layout="wide")

DARK_CSS = """
<style>
:root {
  --bg: #0b1020;
  --panel: #0f1630;
  --text: #dbe7ff;
  --muted: #96a2c2;
  --accent: #3478ff;
  --ok: #00d3a7;
  --warn: #ffb020;
  --bad: #ff5d6c;
  --radius: 18px;
}
.stApp { background: radial-gradient(60% 80% at 20% 0%, #0e1540 0%, #0b1020 60%) fixed; color: var(--text); }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1630, #0b1230);
  border-right: 1px solid rgba(255,255,255,0.05);
}
.neon-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
}
.badge {display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.75rem;}
.badge.ok {background:rgba(0,211,167,0.15); color:#00d3a7;}
.badge.mid {background:rgba(255,176,32,0.15); color:#ffb020;}
.badge.bad {background:rgba(255,93,108,0.15); color:#ff5d6c;}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ============================
# CONSTANTS / PATH
# ============================
FIXED_PATH = r"C:\\Users\\Thomas\\Desktop\\webapp\\tables\\adp_app_table.csv"
REQUIRED_COLS = [
    "Rank","Player","Team","Bye","POS",
    "ESPN","Sleeper","NFL","RTSports","FFC","Fantrax",
    "AVG","FPTS"
]

# ============================
# UTILS
# ============================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def survival_prob(adp: float, std: float, current_overall: int, next_overall: int) -> float:
    if std is None or std <= 0: return float("nan")
    z_next = (next_overall - adp) / std
    z_curr = (current_overall - adp) / std
    p_between = norm_cdf(z_next) - norm_cdf(z_curr)
    return max(0.0, min(1.0, 1 - p_between))

def next_pick_overall(num_teams: int, my_slot: int, current_overall: int, snake: bool=True) -> int:
    if not snake:
        picks_ahead = (my_slot - 1) - (current_overall % num_teams)
        if picks_ahead <= 0: picks_ahead += num_teams
        return current_overall + picks_ahead
    for step in range(1, 3*num_teams + 10):
        candidate = current_overall + step
        round_num = (candidate - 1) // num_teams + 1
        pos_in_round = ((candidate - 1) % num_teams) + 1
        my_pos_in_round = my_slot if (round_num % 2 == 1) else (num_teams - my_slot + 1)
        if pos_in_round == my_pos_in_round:
            return candidate
    return current_overall + num_teams

def next_picks_overall(num_teams: int, my_slot: int, current_overall: int, snake: bool=True, k: int=2) -> List[int]:
    picks, last = [], current_overall
    for _ in range(k):
        nxt = next_pick_overall(num_teams, my_slot, last, snake)
        picks.append(nxt); last = nxt
    return picks

# ============================
# DATA LOADING (IGNORE adp_std)
# ============================
@st.cache_data
def load_fixed_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")
    df = df[REQUIRED_COLS].copy()
    # Normalize
    num_cols = ["Rank","Bye","ESPN","Sleeper","NFL","RTSports","FFC","Fantrax","AVG","FPTS"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["POS"] = df["POS"].astype(str).str.upper().str.strip()
    return df

# Empirical STD solely from sources
SOURCES = ["ESPN","Sleeper","NFL","RTSports","FFC","Fantrax"]
POS_STD_DEFAULTS = {"QB": 9.0, "RB": 12.0, "WR": 12.0, "TE": 10.0}

@st.cache_data
def compute_empirical_std(df: pd.DataFrame) -> pd.Series:
    # std across sources (ddof=1)
    vals = df[SOURCES].astype(float)
    emp = vals.std(axis=1, ddof=1)
    return emp

def fill_std_with_pos_fallback(emp_std: pd.Series, pos_series: pd.Series, fallback_scale: float) -> pd.Series:
    out = emp_std.copy()
    for i, (e, pos) in enumerate(zip(emp_std, pos_series)):
        if pd.isna(e) or e == 0:
            default = POS_STD_DEFAULTS.get(pos, 10.0)
            out.iat[i] = default * fallback_scale
    return out

# ============================
# SIDEBAR
# ============================
st.sidebar.title("🏈 Draft Assistant")
num_teams = st.sidebar.number_input("Times", 4, 20, 12)
my_slot = st.sidebar.number_input("Seu slot", 1, 20, 5)
snake = st.sidebar.checkbox("Snake", True)

st.sidebar.subheader("Desvio padrão (apenas fontes)")
fallback_scale = st.sidebar.slider("Fator fallback por posição", 0.2, 2.0, 1.0, 0.1)

# ============================
# LOAD DATA
# ============================
try:
    df = load_fixed_table(FIXED_PATH)
except Exception as e:
    st.error(f"Erro ao abrir arquivo: {e}")
    st.stop()

# STD from sources only + fallback by position
emp_std = compute_empirical_std(df)
df["STD_final"] = fill_std_with_pos_fallback(emp_std, df["POS"], fallback_scale)
df["ADP_base"] = df["AVG"]

# ============================
# MAIN TABS
# ============================
tab_liga, tab_dados, tab_live = st.tabs(["⚙️ Liga", "🧾 Dados", "🚦 Draft Live"])

with tab_liga:
    st.markdown("### Configuração")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Times", num_teams)
        st.metric("Seu slot", my_slot)
        st.metric("Formato", "Snake" if snake else "Linear")
    with c2:
        st.metric("Fallback POS scale", fallback_scale)
    st.caption("STD calculado **exclusivamente** a partir das fontes ESPN/Sleeper/NFL/RTSports/FFC/Fantrax. Se não houver variação suficiente, aplico fallback por posição.")

with tab_dados:
    st.markdown("### Tabela de dados (somente colunas autorizadas)")
    st.dataframe(df, use_container_width=True, height=550)

with tab_live:
    if "picked" not in st.session_state: st.session_state.picked = []
    if "current_overall" not in st.session_state: st.session_state.current_overall = 0

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.session_state.current_overall = st.number_input("Pick atual (já passada)", 0, 400, st.session_state.current_overall, 1)
    with c2:
        pick_name = st.selectbox("Quem saiu agora?", ["(selecione)"] + df["Player"].tolist())
        if st.button("Marcar saída"):
            if pick_name != "(selecione)" and pick_name not in st.session_state.picked:
                st.session_state.picked.append(pick_name)
                st.session_state.current_overall += 1
    with c3:
        if st.button("↩️ Desfazer última"):
            if st.session_state.picked:
                st.session_state.picked.pop()
                st.session_state.current_overall = max(0, st.session_state.current_overall - 1)
        if st.button("🧹 Limpar marcados"):
            st.session_state.picked = []

    st.markdown("---")

    # Available / picked
    picked_df = df[df["Player"].isin(st.session_state.picked)]
    avail_df = df[~df["Player"].isin(st.session_state.picked)].copy()

    # Simple expected/observed mix (for future room bias hooks)
    exp_mix = avail_df["POS"].value_counts(normalize=True).to_dict()
    obs_mix = picked_df["POS"].value_counts(normalize=True).to_dict() if not picked_df.empty else {}

    # Next picks
    def next_picks_overall_local():
        return next_picks_overall(num_teams, my_slot, st.session_state.current_overall, snake, k=2)

    picks = next_picks_overall_local()
    st.markdown(f"**Suas próximas picks gerais:** `{picks}`")

    # Probabilities (using ADP_base and STD_final)
    avail_df["Prob_next1"] = avail_df.apply(lambda r: survival_prob(r["ADP_base"], r["STD_final"], st.session_state.current_overall, picks[0]), axis=1)
    avail_df["Prob_next2"] = avail_df.apply(lambda r: survival_prob(r["ADP_base"], r["STD_final"], st.session_state.current_overall, picks[1]), axis=1)

    # Replacement levels (quick): last starter per league
    st.subheader("Titulares por time (replacement)")
    col_qb, col_rb, col_wr, col_te = st.columns(4)
    with col_qb:
        qb_s = st.number_input("QB", 0, 3, 1)
    with col_rb:
        rb_s = st.number_input("RB", 0, 5, 2)
    with col_wr:
        wr_s = st.number_input("WR", 0, 5, 2)
    with col_te:
        te_s = st.number_input("TE", 0, 3, 1)
    starters = {"QB": qb_s, "RB": rb_s, "WR": wr_s, "TE": te_s}

    def compute_replacement(df_all: pd.DataFrame, starters_map: Dict[str,int], teams: int) -> Dict[str,float]:
        repl = {}
        for pos, k in starters_map.items():
            n = max(1, k * teams)
            pool = df_all[df_all["POS"] == pos].sort_values("FPTS", ascending=False)
            if len(pool) >= n:
                repl[pos] = float(pool.iloc[n-1]["FPTS"])
            elif len(pool):
                repl[pos] = float(pool["FPTS"].min())
            else:
                repl[pos] = 0.0
        return repl

    repl_levels = compute_replacement(df, starters, num_teams)
    avail_df["VM"] = avail_df["FPTS"] - avail_df["POS"].map(repl_levels).fillna(0.0)

    # Recommendations per position
    st.subheader("Recomendações por posição (pular agora vs melhor alternativa provável)")
    prob_thresh = st.slider("Limiar de prob. para sobrar (próxima pick)", 0.0, 1.0, 0.6, 0.05)
    top_n = st.slider("Top por posição", 3, 15, 5, 1)

    rows = []
    for pos, g in avail_df.groupby("POS"):
        g2 = g.sort_values(["FPTS","Prob_next1"], ascending=[False, False]).copy()
        for _, row in g2.iterrows():
            pool = g2[(g2["Player"] != row["Player"]) & (g2["Prob_next1"] >= prob_thresh)]
            alt = pool.sort_values("FPTS", ascending=False).head(1)
            alt_name = alt["Player"].iat[0] if not alt.empty else None
            alt_pts = float(alt["FPTS"].iat[0]) if not alt.empty else np.nan
            impact = float(row["FPTS"] - alt_pts) if not np.isnan(alt_pts) else np.nan
            rows.append({
                "Player": row["Player"],
                "POS": pos,
                "Team": row["Team"],
                "FPTS": row["FPTS"],
                "ADP": row["ADP_base"],
                "STD": row["STD_final"],
                "Prob_next1": row["Prob_next1"],
                "Prob_next2": row["Prob_next2"],
                "BestAlt@next1": alt_name,
                "AltFPTS@next1": alt_pts,
                "Impact_if_Skip@next1": impact,
                "VM": row["VM"],
            })

    table = pd.DataFrame(rows)
    recs = table.sort_values(["POS","Impact_if_Skip@next1","FPTS"], ascending=[True, False, False]).groupby("POS").head(top_n)
    st.dataframe(recs.reset_index(drop=True), use_container_width=True, height=360)

    # Global targets
    st.subheader("Top alvos globais (Impacto × risco de sumir)")
    global_df = table.copy()
    global_df["RiskImpact"] = global_df["Impact_if_Skip@next1"] * (1.0 - global_df["Prob_next1"].clip(0,1))
    st.dataframe(global_df.sort_values(["RiskImpact","FPTS"], ascending=[False, False]).head(20).reset_index(drop=True), use_container_width=True, height=360)

    # Quick board
    st.subheader("Board rápido por posição")
    pos_choice = st.multiselect("Filtrar POS", options=sorted(df["POS"].unique().tolist()), default=[])
    board = avail_df.copy()
    if pos_choice:
        board = board[board["POS"].isin(pos_choice)]
    board = board[["Player","Team","POS","FPTS","ADP_base","Prob_next1","Prob_next2"]].copy().sort_values(["POS","FPTS"], ascending=[True, False]).head(80)

    def prob_badge(p):
        if pd.isna(p): return '<span class="badge">?</span>'
        if p >= 0.75: return '<span class="badge ok">↑ 75%</span>'
        if p >= 0.45: return '<span class="badge mid">~ 50%</span>'
        return '<span class="badge bad">↓ 25%</span>'

    def row_html(r):
        return f"<div class='neon-card'><b>{r.Player}</b> <span class='small'>({r.Team} · {r.POS})</span> " \
               f"{prob_badge(r.Prob_next1)}" \
               f"<div class='small'>FPTS: {r.FPTS:.1f} • ADP: {r.ADP_base:.1f}</div></div>"

    st.markdown("".join(row_html(x) for x in board.itertuples()), unsafe_allow_html=True)

    with st.expander("Notas"):
        st.caption("STD calculado **apenas** a partir de ESPN/Sleeper/NFL/RTSports/FFC/Fantrax (ddof=1). Se faltar variação, aplico fallback por POS (QB/RB/WR/TE) escalado pelo fator acima. Probabilidades usam Normal(AVG, STD).")
