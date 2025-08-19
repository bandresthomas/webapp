import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from math import erf, sqrt

# =============================
# ---------- Helpers ----------
# =============================

def normal_cdf(z):
    """Vectorized CDF of the standard normal. Accepts scalars or numpy arrays."""
    z_arr = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(z_arr / np.sqrt(2.0)))


def prob_available_vector(mu_adj: np.ndarray, sigma_adj: np.ndarray, K: int) -> np.ndarray:
    z = (K - mu_adj) / sigma_adj
    return np.clip(1 - normal_cdf(z), 0.0, 1.0)

# =============================
# ---------- Draft Types ------
# =============================

@dataclass
class RosterRules:
    slots: Dict[str, int]     # {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1}
    flex_map: Dict[str, float]  # {"RB":1,"WR":1,"TE":1}


@dataclass
class TeamState:
    taken: Dict[str, int] = field(default_factory=lambda: {"QB": 0, "RB": 0, "WR": 0, "TE": 0})
    players: List[Tuple[str, str]] = field(default_factory=list)  # (player_name, pos)


@dataclass
class DraftState:
    N: int
    user_slot: int           # 1..N
    current_pick: int        # global pick (1-indexed)
    order_snake: bool = True
    roster_rules: RosterRules = None
    teams: List[TeamState] = None
    history: List[Tuple[int, str, str]] = field(default_factory=list)  # (pick_no, player_id, POS)

    def next_pick_for_slot(self, slot: int, from_pick: int = None) -> int:
        if from_pick is None:
            from_pick = self.current_pick
        r = ((from_pick - 1) // self.N) + 1
        p = from_pick
        while True:
            if r % 2 == 1:
                k = (r - 1) * self.N + slot
            else:
                k = (r - 1) * self.N + (self.N - slot + 1)
            if k > p:
                return k
            r += 1

    def user_next_pick(self) -> int:
        return self.next_pick_for_slot(self.user_slot, self.current_pick)

# =============================
# ---------- Run/Needs --------
# =============================

def compute_run_excess(draft: DraftState, window: int, baseline_rate: Dict[str, float]) -> Dict[str, float]:
    recent = draft.history[-window:]
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for _, _, pos in recent:
        if pos in counts:
            counts[pos] += 1
    n = len(recent)
    if n == 0:
        return {p: 0.0 for p in counts}
    n_to_user = draft.user_next_pick() - draft.current_pick
    excess = {}
    for p in counts:
        exp = baseline_rate.get(p, 0.0) * n
        e = max(0.0, counts[p] - exp)
        # project to the interval until the user pick
        excess[p] = e * (n_to_user / max(1, n))
    return excess


def need_scores_until_user(draft: DraftState, players_alive_df: pd.DataFrame) -> Dict[str, float]:
    K_user = draft.user_next_pick()
    teams_to_pick = []
    k = draft.current_pick
    while k < K_user:
        r = ((k - 1) // draft.N) + 1
        if r % 2 == 1:
            slot = ((k - 1) % draft.N) + 1
        else:
            slot = draft.N - ((k - 1) % draft.N)
        teams_to_pick.append(slot - 1)
        k += 1

    pos_list = ["QB", "RB", "WR", "TE"]
    demand = {p: 0.0 for p in pos_list}

    # optional: best tier alive per pos (smaller = better)
    best_by_pos_alive = {
        p: (players_alive_df.loc[players_alive_df["POS"] == p, "tier"].min() if "tier" in players_alive_df.columns else None)
        for p in pos_list
    }

    for idx in teams_to_pick:
        team = draft.teams[idx]
        need_raw = {}
        for p in pos_list:
            max_slots = draft.roster_rules.slots.get(p, 0)
            taken = team.taken.get(p, 0)
            gap = max(max_slots - taken, 0)
            # FLEX share
            flex_room = draft.roster_rules.slots.get("FLEX", 0)
            if flex_room > 0 and p in draft.roster_rules.flex_map:
                if taken < max_slots:
                    gap += 0.5 * flex_room
            need_raw[p] = gap

        # tier weighting
        w_tier = {}
        for p in pos_list:
            if best_by_pos_alive[p] is None:
                w_tier[p] = 0.0
            else:
                w_tier[p] = 1.0 / (best_by_pos_alive[p] + 1.0)

        w_slots, wtier = 1.0, 0.5
        score = {p: w_slots * need_raw[p] + wtier * w_tier[p] for p in pos_list}
        svals = np.array(list(score.values()))
        if float(svals.max()) == 0.0:
            probs = np.ones_like(svals) / len(svals)
        else:
            exps = np.exp(svals - svals.max())
            probs = exps / exps.sum()
        for j, p in enumerate(pos_list):
            demand[p] += float(probs[j])

    return demand


# =============================
# ------- Core Calculator -----
# =============================

def compute_probabilities(players_df: pd.DataFrame,
                          draft: DraftState,
                          chaos: float = 0.3,
                          sigma_min: float = 2.0,
                          gamma_by_pos: Dict[str, float] = {"RB": 1.0, "WR": 0.9, "QB": 0.7, "TE": 1.1},
                          run_window: int = 12,
                          baseline_rate: Dict[str, float] = {"RB": 0.33, "WR": 0.42, "QB": 0.13, "TE": 0.12},
                          tau_cold: float = 0.0,
                          tau_hot: float = 4.0,
                          run_z_threshold: float = 1.0) -> pd.DataFrame:

    alive = players_df[~players_df["picked"].astype(bool)].copy()

    demand_need = need_scores_until_user(draft, alive)  # expected picks by pos until user
    demand_run = compute_run_excess(draft, run_window, baseline_rate)  # run projection

    K_user = draft.user_next_pick()
    pos_list = ["QB", "RB", "WR", "TE"]

    # hot positions inflate variance
    recent = draft.history[-run_window:]
    n = len(recent)
    hot = {p: False for p in pos_list}
    if n > 0:
        counts = {p: 0 for p in pos_list}
        for _, _, p in recent:
            if p in counts:
                counts[p] += 1
        for p in pos_list:
            exp = baseline_rate.get(p, 0.0) * n
            var = n * baseline_rate.get(p, 0.0) * (1 - baseline_rate.get(p, 0.0)) + 1e-6
            z = (counts[p] - exp) / sqrt(var)
            hot[p] = z >= run_z_threshold

    mu = alive["ADP"].to_numpy(float)
    sig = alive["ADP_STD"].to_numpy(float)
    pos = alive["POS"].to_numpy(str)

    mu_adj = mu.copy()
    sigma_adj = np.maximum(sig * (1 + chaos), sigma_min)

    tau_by_pos = {p: (tau_hot if hot[p] else tau_cold) for p in pos_list}

    for i in range(len(alive)):
        p = pos[i]
        shift = gamma_by_pos.get(p, 0.8) * (demand_need.get(p, 0.0) + demand_run.get(p, 0.0))
        mu_adj[i] = mu[i] - shift
        sigma_adj[i] = sqrt(sigma_adj[i] ** 2 + tau_by_pos.get(p, 0.0) ** 2)

    probs = prob_available_vector(mu_adj, sigma_adj, K_user)

    out = alive.copy()
    out["mu_adj"] = mu_adj
    out["sigma_adj"] = sigma_adj
    out["prob_available_next_pick"] = probs
    out.sort_values("prob_available_next_pick", ascending=False, inplace=True)

    return out[["player_id", "Player", "POS", "ADP", "ADP_STD", "mu_adj", "sigma_adj", "prob_available_next_pick"]]


# =============================
# --------- Data Loading ------
# =============================

def load_players(default_path: str = None, uploaded: bytes = None) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if default_path and os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            raise FileNotFoundError("Arquivo de jogadores não encontrado. Forneça o CSV.")

    # Expect columns: Rank, Player, Team, Bye, POS, ESPN, Sleeper, NFL, RTSports, FFC, Fantrax, AVG, adp_std, FPTS
    source_cols = ["ESPN", "Sleeper", "NFL", "RTSports", "FFC", "Fantrax"]
    for c in source_cols:
        if c not in df.columns:
            st.warning(f"Coluna '{c}' não encontrada. O ADP será calculado só com as colunas disponíveis.")
    present_sources = [c for c in source_cols if c in df.columns]

    for c in present_sources:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if "AVG" in df.columns:
        df["ADP"] = pd.to_numeric(df["AVG"], errors='coerce')
    else:
        df["ADP"] = df[present_sources].mean(axis=1, skipna=True)

    if "adp_std" in df.columns:
        df["ADP_STD"] = pd.to_numeric(df["adp_std"], errors='coerce')
    else:
        df["ADP_STD"] = df[present_sources].std(axis=1, ddof=0, skipna=True).fillna(5.0)

    if "player_id" not in df.columns:
        df["player_id"] = (
            df.get("Player", pd.Series(range(len(df)))).astype(str).str.lower().str.replace(" ", "_", regex=False)
            + "_" + df.get("Team", pd.Series("")).astype(str).str.lower()
        )

    if "POS" not in df.columns:
        raise ValueError("A coluna POS é obrigatória no CSV (QB/RB/WR/TE).")

    if "tier" not in df.columns:
        df["tier"] = 999

    df["picked"] = False
    return df[["player_id", "Player", "Team", "POS", "ADP", "ADP_STD", "tier", "picked"]]


# =============================
# ------------- UI ------------
# =============================

st.set_page_config(page_title="ADP Prob Draft", layout="wide")

st.title("🔮 Probabilidade de Jogador Disponível — Draft NFL (com runs e necessidades)")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Configurações da Liga")
    N = st.number_input("Nº de times", min_value=4, max_value=16, value=10, step=1)
    user_slot = st.number_input("Seu slot (1=Nº times)", min_value=1, max_value=int(N), value=6, step=1)

    st.markdown("---")
    st.header("Parâmetros do Modelo")
    chaos = st.slider("Caos (imprevisibilidade)", 0.0, 0.8, 0.3, 0.05)
    run_window = st.slider("Janela de run (últimas picks)", 6, 20, 12, 1)
    sigma_min = st.slider("Desvio mínimo (picks)", 0.5, 6.0, 2.0, 0.5)
    tau_hot = st.slider("Inflar variância quando em run (τ)", 0.0, 8.0, 4.0, 0.5)
    run_z_threshold = st.slider("Z-score p/ marcar run", 0.5, 2.5, 1.0, 0.1)

    st.markdown("---")
    st.header("Dados")
    default_path = r"C:\\Users\\Thomas\\Desktop\\webapp\\tables\\adp_app_table.csv"
    st.caption(f"Caminho padrão: {default_path}")
    uploaded = st.file_uploader("Ou envie o CSV aqui", type=["csv"])  # opcional para testar

# ---------- Init/Load Data ----------
if "players_df" not in st.session_state:
    try:
        st.session_state.players_df = load_players(default_path=default_path, uploaded=uploaded)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

# allow reload if user uploads after start
if uploaded is not None and st.session_state.get("_uploaded_sha") != hash(uploaded.getvalue()):
    try:
        st.session_state.players_df = load_players(uploaded=uploaded)
        st.session_state["_uploaded_sha"] = hash(uploaded.getvalue())
        for key in ["draft", "pick_log"]:
            st.session_state.pop(key, None)
        st.rerun()
    except Exception as e:
        st.error(f"Erro ao recarregar dados enviados: {e}")
        st.stop()

players_df = st.session_state.players_df

# ---------- Initialize Draft State ----------
if "draft" not in st.session_state or st.session_state.draft.N != N or st.session_state.draft.user_slot != user_slot:
    teams = [TeamState() for _ in range(int(N))]
    roster = RosterRules(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, flex_map={"RB": 1, "WR": 1, "TE": 1})
    st.session_state.draft = DraftState(N=int(N), user_slot=int(user_slot), current_pick=1, roster_rules=roster, teams=teams)
    st.session_state.pick_log = []  # for undo

draft = st.session_state.draft

# ---------- Controls: Draft Picks (busca rápida) ----------
col_pick, col_info = st.columns([1, 2])
with col_pick:
    r = ((draft.current_pick - 1) // draft.N) + 1
    if r % 2 == 1:
        on_clock_slot = ((draft.current_pick - 1) % draft.N) + 1
    else:
        on_clock_slot = draft.N - ((draft.current_pick - 1) % draft.N)

    st.subheader(f"🕒 Agora: Pick #{draft.current_pick} — Time {on_clock_slot}")

    alive = players_df[~players_df["picked"].astype(bool)].copy()

    q = st.text_input("Buscar jogador")
    pos_filter = st.multiselect("Filtrar posições", ["QB", "RB", "WR", "TE"], default=["RB", "WR", "QB", "TE"])
    if q:
        alive = alive[alive["Player"].str.contains(q, case=False, na=False)]
    if pos_filter:
        alive = alive[alive["POS"].isin(pos_filter)]

    # compute probabilities preview for selector
    preview_df = players_df.copy()
    probs_preview = compute_probabilities(
        preview_df,
        draft,
        chaos=chaos,
        sigma_min=sigma_min,
        run_window=run_window,
        tau_hot=tau_hot,
        run_z_threshold=run_z_threshold,
    )
    preview_small = probs_preview.set_index("player_id")["prob_available_next_pick"]

    alive = alive.copy()
    alive["prob"] = alive["player_id"].map(preview_small)
    alive.sort_values(["prob", "ADP"], ascending=[False, True], inplace=True)

    options = alive.head(300)[["player_id", "Player", "POS", "Team", "ADP", "prob"]]
    options_display = options.apply(lambda r: f"{r['Player']} ({r['POS']}-{r['Team']}) • ADP {int(r['ADP']) if pd.notna(r['ADP']) else '-'} • P(sobrar) {r['prob']:.2f}", axis=1)
    label_to_id = dict(zip(options_display, options["player_id"]))

    selected_label = st.selectbox("Selecionar (busca rápida)", list(label_to_id.keys()) if len(label_to_id) > 0 else ["Sem opções"])

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("✅ Draftar (busca)") and selected_label in label_to_id:
        pid = label_to_id[selected_label]
        row = players_df.loc[players_df["player_id"] == pid].iloc[0]
        players_df.loc[players_df["player_id"] == pid, "picked"] = True
        draft.history.append((draft.current_pick, pid, row["POS"]))
        draft.teams[on_clock_slot - 1].players.append((row["Player"], row["POS"]))
        draft.teams[on_clock_slot - 1].taken[row["POS"]] += 1
        st.session_state.pick_log.append(pid)
        draft.current_pick += 1
        st.rerun()

    if col_btn2.button("↩️ Desfazer última pick"):
        if st.session_state.pick_log:
            last_pid = st.session_state.pick_log.pop()
            draft.current_pick = max(1, draft.current_pick - 1)
            r_back = ((draft.current_pick - 1) // draft.N) + 1
            if r_back % 2 == 1:
                slot_back = ((draft.current_pick - 1) % draft.N) + 1
            else:
                slot_back = draft.N - ((draft.current_pick - 1) % draft.N)
            row = players_df.loc[players_df["player_id"] == last_pid].iloc[0]
            team_players = draft.teams[slot_back - 1].players
            for idx in range(len(team_players) - 1, -1, -1):
                if team_players[idx][0] == row["Player"]:
                    team_players.pop(idx)
                    break
            draft.teams[slot_back - 1].taken[row["POS"]] = max(0, draft.teams[slot_back - 1].taken[row["POS"]] - 1)
            players_df.loc[players_df["player_id"] == last_pid, "picked"] = False
            if draft.history and draft.history[-1][1] == last_pid:
                draft.history.pop()
            st.rerun()

with col_info:
    st.subheader("🎯 Seu próximo pick")
    K_user = draft.user_next_pick()
    st.metric("Pick global", K_user)

    result_df = compute_probabilities(
        players_df.copy(),
        draft,
        chaos=chaos,
        sigma_min=sigma_min,
        run_window=run_window,
        tau_hot=tau_hot,
        run_z_threshold=run_z_threshold,
    )

    # ==== Tabela com seleção direta ====
    st.markdown("**Tabela de probabilidades (vivos):**")

    # aplica filtro de posições (multiselect retorna lista)
    if pos_filter:
        result_df = result_df[result_df["POS"].isin(pos_filter)]

    show_df = result_df.copy()
    show_df["ADP"] = show_df["ADP"].round(1)
    show_df["imprev"] = show_df["ADP_STD"].round(2)
    show_df["ADP_adj"] = show_df["mu_adj"].round(2)
    show_df["caos"] = show_df["sigma_adj"].round(2)
    show_df["Prob próximo pick (%)"] = (show_df["prob_available_next_pick"] * 100).round(0).astype(int)
    show_df = show_df.sort_values(["ADP", "Player"], ascending=[True, True])
    show_df = show_df[["player_id", "Player", "POS", "ADP", "imprev", "ADP_adj", "caos", "Prob próximo pick (%)"]]
    show_df["Selecionar"] = False
    show_df = show_df.set_index("player_id", drop=True)


    edited = st.data_editor(
        show_df,
        use_container_width=True,
        height=520,
        key="table_editor_right",
        hide_index=True,
        column_config={
            "Prob próximo pick (%)": st.column_config.NumberColumn(format="%d%%"),
            "Selecionar": st.column_config.CheckboxColumn(help="Marque o jogador que você quer draftar agora"),
        },
        disabled=["Player", "POS", "ADP", "imprev", "ADP_adj", "caos", "Prob próximo pick (%)"],
    )

    col_tbl_btn1, col_tbl_btn2 = st.columns([1, 1])
    if col_tbl_btn1.button("✅ Draftar selecionado (tabela)"):
        sel_rows = edited[edited["Selecionar"] == True] if isinstance(edited, pd.DataFrame) else pd.DataFrame()
        if not sel_rows.empty:
            pid = sel_rows.index[0]  # usa o INDEX oculto (player_id)
            row = players_df.loc[players_df["player_id"] == pid].iloc[0]
            # quem está no relógio agora
            r_cur = ((draft.current_pick - 1) // draft.N) + 1
            if r_cur % 2 == 1:
                on_slot2 = ((draft.current_pick - 1) % draft.N) + 1
            else:
                on_slot2 = draft.N - ((draft.current_pick - 1) % draft.N)
            # aplica pick
            players_df.loc[players_df["player_id"] == pid, "picked"] = True
            draft.history.append((draft.current_pick, pid, row["POS"]))
            draft.teams[on_slot2 - 1].players.append((row["Player"], row["POS"]))
            draft.teams[on_slot2 - 1].taken[row["POS"]] += 1
            st.session_state.pick_log.append(pid)
            draft.current_pick += 1
            st.rerun()

    if col_tbl_btn2.button("↩️ Desfazer (tabela)"):
        if st.session_state.pick_log:
            last_pid = st.session_state.pick_log.pop()
            draft.current_pick = max(1, draft.current_pick - 1)
            r_back = ((draft.current_pick - 1) // draft.N) + 1
            if r_back % 2 == 1:
                slot_back = ((draft.current_pick - 1) % draft.N) + 1
            else:
                slot_back = draft.N - ((draft.current_pick - 1) % draft.N)
            row = players_df.loc[players_df["player_id"] == last_pid].iloc[0]
            team_players = draft.teams[slot_back - 1].players
            for idx in range(len(team_players) - 1, -1, -1):
                if team_players[idx][0] == row["Player"]:
                    team_players.pop(idx)
                    break
            draft.teams[slot_back - 1].taken[row["POS"]] = max(0, draft.teams[slot_back - 1].taken[row["POS"]] - 1)
            players_df.loc[players_df["player_id"] == last_pid, "picked"] = False
            if draft.history and draft.history[-1][1] == last_pid:
                draft.history.pop()
            st.rerun()

# ---------- Board (Times x Rounds) ----------
st.markdown("---")
st.subheader("📋 Board (Times x Rounds)")

max_picks = max(draft.current_pick - 1, 0)
rounds_completed = ((max_picks) // draft.N) + (1 if (max_picks % draft.N) else 0)
rounds_to_show = max(8, rounds_completed + 1)

cols = st.columns(int(draft.N))
for i in range(int(draft.N)):
    with cols[i]:
        st.markdown(f"**Time {i+1}**")
        team_players = draft.teams[i].players
        for r in range(rounds_to_show):
            if r < len(team_players):
                nm, ps = team_players[r]
                st.markdown(f"R{r+1}: **{nm}** ({ps})")
            else:
                st.markdown(f"R{r+1}: —")

st.markdown(
    "<small>Coloque o arquivo `adp_app_table.csv` no caminho padrão (ou faça upload na barra lateral). O app recalcula as probabilidades a cada pick conforme runs e necessidades dos adversários.</small>",
    unsafe_allow_html=True,
)