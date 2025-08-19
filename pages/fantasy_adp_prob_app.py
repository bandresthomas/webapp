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


def prob_available_vector(mu: np.ndarray, sigma: np.ndarray, K: int) -> np.ndarray:
    """P(X > K) for X ~ N(mu, sigma^2). Clipped to [0,1]. Safe for arrays."""
    sigma_safe = np.where(~np.isfinite(sigma) | (sigma <= 0), 1.0, sigma)
    z = (K - mu) / sigma_safe
    p = 1.0 - normal_cdf(z)
    return np.clip(p, 0.0, 1.0)


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
# -------- Data Hygiene -------
# =============================

def sanitize_player_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize common column names
    df.rename(columns={"adp_std": "ADP_STD", "avg": "AVG", "fpts": "FPTS"}, inplace=True)

    # POS validation
    if "POS" not in df.columns:
        raise ValueError("A coluna POS é obrigatória (QB/RB/WR/TE).")
    df["POS"] = df["POS"].astype(str).str.upper().str.strip()
    valid_pos = {"QB", "RB", "WR", "TE"}
    df = df[df["POS"].isin(valid_pos)]

    # Ensure numeric
    for c in ["ADP", "ADP_STD", "FPTS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If ADP missing but sources exist, compute
    source_cols = [c for c in ["ESPN","Sleeper","NFL","RTSports","FFC","Fantrax"] if c in df.columns]
    for c in source_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ADP" not in df.columns and source_cols:
        df["ADP"] = df[source_cols].mean(axis=1, skipna=True)
    if "ADP_STD" not in df.columns and source_cols:
        df["ADP_STD"] = df[source_cols].std(axis=1, ddof=0, skipna=True)

    # Remove non-finite ADP/STD
    for c in ["ADP","ADP_STD"]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df[df["ADP"].notna() & df["ADP_STD"].notna()]

    # Practical caps
    df["ADP"] = df["ADP"].clip(lower=1)
    df["ADP_STD"] = df["ADP_STD"].clip(lower=0.5)

    # Minimum columns
    if "Player" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'Player'.")
    if "Team" not in df.columns:
        df["Team"] = ""
    if "FPTS" not in df.columns:
        df["FPTS"] = np.nan  # optional
    if "tier" not in df.columns:
        df["tier"] = np.nan  # optional

    if "player_id" not in df.columns:
        df["player_id"] = (
            df["Player"].astype(str).str.lower().str.replace(" ", "_", regex=False)
            + "_" + df["Team"].astype(str).str.lower()
        )

    return df


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
                          tau_hot: float = 3.0,
                          run_z_threshold: float = 1.0) -> pd.DataFrame:
    """Simplified, robust probability computation.
    - Filters invalid rows
    - Light run detection inflates variance for hot positions
    - Needs estimate: counts how many picks before user and distributes by baseline
    """
    alive = players_df.loc[~players_df["picked"].astype(bool)].copy()
    alive = alive[np.isfinite(alive["ADP"]) & np.isfinite(alive["ADP_STD"])]
    if alive.empty:
        return alive.assign(mu_adj=[], sigma_adj=[], prob_available_next_pick=[])

    # How many picks until user?
    K_user = draft.user_next_pick()
    picks_to_user = max(0, K_user - draft.current_pick)

    # Baseline need per position scaled to picks_to_user
    pos_list = ["QB","RB","WR","TE"]
    need_counts = {p: baseline_rate.get(p, 0.25) * picks_to_user for p in pos_list}

    # Simple run detection on recent picks
    recent = draft.history[-run_window:]
    if len(recent) > 0:
        counts = {p:0 for p in pos_list}
        for _,_,p in recent:
            if p in counts:
                counts[p]+=1
        n = len(recent)
        hot = {p: ((counts[p] - baseline_rate.get(p,0)*n) / max(1e-6, math.sqrt(n*baseline_rate.get(p,0)*(1-baseline_rate.get(p,0))+1e-6)) >= run_z_threshold) for p in pos_list}
    else:
        hot = {p: False for p in pos_list}

    mu = alive["ADP"].to_numpy(float)
    sig = alive["ADP_STD"].to_numpy(float)
    pos = alive["POS"].to_numpy(str)

    mu_adj = mu.copy()
    sigma_adj = np.maximum(sig * (1 + chaos), sigma_min)

    # Shift mean earlier for positions with higher expected demand
    demand_by_pos = {p: need_counts.get(p,0.0) for p in pos_list}
    for i in range(len(alive)):
        p = pos[i]
        shift = gamma_by_pos.get(p,1.0) * demand_by_pos.get(p,0.0) / max(1, len(pos_list))
        mu_adj[i] = mu[i] - shift
        if hot.get(p, False):
            sigma_adj[i] = np.sqrt(sigma_adj[i]**2 + tau_hot**2)

    probs = prob_available_vector(mu_adj, sigma_adj, K_user)

    out = alive.copy()
    out["mu_adj"] = mu_adj
    out["sigma_adj"] = sigma_adj
    out["prob_available_next_pick"] = probs
    return out


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

    # If AVG/adp_std present, map to ADP/ADP_STD
    if "AVG" in df.columns:
        df["ADP"] = pd.to_numeric(df["AVG"], errors="coerce")
    source_cols = [c for c in ["ESPN","Sleeper","NFL","RTSports","FFC","Fantrax"] if c in df.columns]
    if "ADP" not in df.columns and source_cols:
        df["ADP"] = df[source_cols].mean(axis=1, skipna=True)
    if "adp_std" in df.columns:
        df["ADP_STD"] = pd.to_numeric(df["adp_std"], errors="coerce")
    elif source_cols:
        df["ADP_STD"] = df[source_cols].std(axis=1, ddof=0, skipna=True)

    df = sanitize_player_data(df)
    df["picked"] = False
    return df[["player_id","Player","Team","POS","ADP","ADP_STD","FPTS","tier","picked"]]


# =============================
# ----- Card calc + UI --------
# =============================

def compute_card_for_pos(result_df: pd.DataFrame, pos: str) -> Dict[str, str]:
    dpos = result_df[result_df["POS"] == pos].copy()
    if dpos.empty:
        return {"main":"—","psobrar":"—","fpts1":"—","next_name":"—","fpts2":"—","custo":"—","risk":"—"}

    # Choose top by FPTS when available, else by ADP asc
    if "FPTS" in dpos.columns and dpos["FPTS"].notna().any():
        dpos = dpos.sort_values(["FPTS","ADP"], ascending=[False, True])
    else:
        dpos = dpos.sort_values(["ADP","Player"], ascending=[True, True])

    j1 = dpos.iloc[0]
    name1 = str(j1["Player"])  # full name
    p1 = float(j1.get("prob_available_next_pick", np.nan))
    fpts1 = float(j1.get("FPTS", np.nan))

    # próximo melhor (exige P(sobrar) >= 50%); se não houver, usa fallback do próximo da lista
    next_thresh = 0.50
    if len(dpos) > 1:
        dpos_rest = dpos.iloc[1:].copy()
        # coluna de prob pode não existir/vir NaN; normaliza
        if "prob_available_next_pick" in dpos_rest.columns:
            dpos_rest["p_next"] = pd.to_numeric(dpos_rest["prob_available_next_pick"], errors="coerce").fillna(0.0)
        else:
            dpos_rest["p_next"] = 0.0

        cands = dpos_rest[dpos_rest["p_next"] >= next_thresh]
        if not cands.empty:
            if "FPTS" in cands.columns and cands["FPTS"].notna().any():
                cands = cands.sort_values(["FPTS", "ADP"], ascending=[False, True])
            else:
                cands = cands.sort_values(["ADP", "Player"], ascending=[True, True])
            j2 = cands.iloc[0]
        else:
            # fallback: usa o próximo da lista original
            j2 = dpos_rest.iloc[0]
        fpts2 = float(j2["FPTS"]) if "FPTS" in dpos.columns and pd.notna(j2["FPTS"]) else (fpts1 if np.isfinite(fpts1) else np.nan)
        name2 = str(j2["Player"])  # nome completo
    else:
        j2 = None
        fpts2 = 0.0 if np.isfinite(fpts1) else np.nan
        name2 = "—"


    # Formats
    if np.isfinite(p1):
        ps = int(round(np.clip(p1,0,1)*100))
        psobrar_txt = f"{ps}%"
    else:
        psobrar_txt = "N/A"

    fpts1_txt = f"{fpts1:.1f}" if np.isfinite(fpts1) else "N/A"
    fpts2_txt = f"{fpts2:.1f}" if np.isfinite(fpts2) else "N/A"

    # EV if pass ≈ p1*FPTS1 + (1-p1)*FPTS2
    if np.isfinite(p1) and np.isfinite(fpts1) and np.isfinite(fpts2):
        ev_pass = p1*fpts1 + (1-p1)*fpts2
        custo = fpts1 - ev_pass
        custo_txt = f"{custo:.1f} pts"
    else:
        custo_txt = "N/A"

    # Risk missing tier: among same tier as j1
    tier1 = j1.get("tier", np.nan)
    if np.isfinite(tier1):
        same_tier = dpos[dpos.get("tier").astype(float) == float(tier1)]
        if not same_tier.empty and "prob_available_next_pick" in same_tier.columns:
            probs = same_tier["prob_available_next_pick"].astype(float).clip(0,1)
            miss = float(np.prod(1 - probs.values))
            risk_txt = f"{int(round(miss*100))}%"
        else:
            risk_txt = "—"
    else:
        risk_txt = "N/A"

    return {"main":name1, "psobrar":psobrar_txt, "fpts1":fpts1_txt, "next_name":name2, "fpts2":fpts2_txt, "custo":custo_txt, "risk":risk_txt}


def render_cards(cards: Dict[str, Dict[str, str]]):
    st.markdown(
        """
        <style>
          .card {border-radius:16px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.06) 0%,rgba(255,255,255,.02) 100%);backdrop-filter:blur(6px);height:100%}
          .card h4{margin:0 0 8px 0;font-size:1.05rem;letter-spacing:.5px;text-transform:uppercase;opacity:.85}
          .card .main{font-weight:700;font-size:1.05rem;margin-bottom:10px}
          .grid{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;font-size:.92rem}
          .kv{opacity:.9}.val{font-weight:600;text-align:right}
          .cost{margin-top:10px;padding-top:10px;border-top:1px dashed rgba(255,255,255,.15);display:flex;justify-content:space-between;font-size:1rem}
        </style>
        """,
        unsafe_allow_html=True,
    )
    labels = {"QB":"QUARTERBACK","RB":"RUNNING BACK","WR":"WIDE RECEIVER","TE":"TIGHT END"}
    cols = st.columns(4)
    for i, pos in enumerate(["QB","RB","WR","TE"]):
        c = cards.get(pos, {})
        with cols[i]:
            st.markdown(f"""
            <div class=card>
              <h4>{labels.get(pos,pos)}</h4>
              <div class=main>{c.get('main','—')}</div>
              <div class=grid>
                <div class=kv>P(sobrar)</div><div class=val>{c.get('psobrar','—')}</div>
                <div class=kv>FPTS Pick</div><div class=val>{c.get('fpts1','—')}</div>
                <div class=kv>Próximo</div><div class=val>{c.get('next_name','—')}</div>
                <div class=kv>FPTS Next</div><div class=val>{c.get('fpts2','—')}</div>
                <div class=kv>Risco Missing Tier</div><div class=val>{c.get('risk','—')}</div>
              </div>
              <div class=cost><div>Custo de Passe</div><div class=val>{c.get('custo','—')}</div></div>
            </div>
            """, unsafe_allow_html=True)


# =============================
# ------------- UI ------------
# =============================

st.set_page_config(page_title="ADP Prob Draft", layout="wide")

st.title("🔮 Probabilidade de Jogador Disponível — Draft NFL (com runs e necessidades)")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Configurações da Liga")
    N = st.number_input("Nº de times", min_value=4, max_value=16, value=10, step=1)
    user_slot = st.number_input("Seu slot (1=Nº times)", min_value=1, max_value=int(N), value=1, step=1)

    st.markdown("---")
    st.header("Parâmetros do Modelo")
    chaos = st.slider("Caos (imprevisibilidade)", 0.0, 0.8, 0.3, 0.05)
    run_window = st.slider("Janela de run (últimas picks)", 6, 20, 12, 1)
    sigma_min = st.slider("Desvio mínimo (picks)", 0.5, 6.0, 2.0, 0.5)
    tau_hot = st.slider("Inflar variância quando em run (τ)", 0.0, 8.0, 3.0, 0.5)
    run_z_threshold = st.slider("Z-score p/ marcar run", 0.5, 2.5, 1.0, 0.1)

    st.markdown("---")
    st.header("Dados")
    default_path = r"C:\Users\Thomas\Desktop\webapp\tables\adp_app_table.csv"
    st.caption(f"Caminho padrão: {default_path}")
    uploaded = st.file_uploader("Ou envie o CSV aqui", type=["csv"])  # opcional

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
    preview_small = probs_preview.set_index("player_id")["prob_available_next_pick"] if not probs_preview.empty else pd.Series(dtype=float)

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

    # ==== Cards de custo por posição ====
    cards = {p: compute_card_for_pos(result_df, p) for p in ["QB","RB","WR","TE"]}
    render_cards(cards)

    # ==== Tabela com seleção direta ====
    st.markdown("**Tabela de probabilidades (vivos):**")

    filtered_df = result_df.copy()
    if pos_filter:
        filtered_df = filtered_df[filtered_df["POS"].isin(pos_filter)]

    show_df = filtered_df.copy()
    if show_df.empty:
        st.info("Sem jogadores para exibir com os filtros atuais.")
    else:
        show_df["ADP"] = pd.to_numeric(show_df["ADP"], errors="coerce").round(1)
        show_df["imprev"] = pd.to_numeric(show_df["ADP_STD"], errors="coerce").round(2)
        show_df["ADP_adj"] = pd.to_numeric(show_df["mu_adj"], errors="coerce").round(2)
        show_df["caos"] = pd.to_numeric(show_df["sigma_adj"], errors="coerce").round(2)
        # Probability percent safe cast
        prob_series = (pd.to_numeric(show_df["prob_available_next_pick"], errors="coerce") * 100).round(0)
        prob_series = prob_series.clip(0, 100).fillna(0).astype(int)
        show_df["Prob próximo pick (%)"] = prob_series
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
                pid = sel_rows.index[0]
                row = players_df.loc[players_df["player_id"] == pid].iloc[0]
                r_cur = ((draft.current_pick - 1) // draft.N) + 1
                if r_cur % 2 == 1:
                    on_slot2 = ((draft.current_pick - 1) % draft.N) + 1
                else:
                    on_slot2 = draft.N - ((draft.current_pick - 1) % draft.N)
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
