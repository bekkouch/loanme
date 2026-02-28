"""
Simulateur de Prêts Immobiliers
================================
Lancer avec : streamlit run simulateur_pret.py

Dépendances :
    pip install streamlit plotly pandas numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulateur Prêts Immobiliers",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #080a10; }
    [data-testid="stSidebar"] { background-color: #0f1117; border-right: 1px solid #1e2130; }
    [data-testid="stSidebar"] * { color: #c8ccd8 !important; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #eef0f6; }
    p, label { color: #c8ccd8; }
    [data-testid="metric-container"] {
        background: #0f1117;
        border: 1px solid #1e2130;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"]  { color: #8b8fa8 !important; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
    [data-testid="stMetricValue"]  { color: #e8c97a !important; font-family: 'DM Mono', monospace; }
    [data-testid="stMetricDelta"]  { font-size: 11px; }
    hr { border-color: #1e2130; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS FINANCIERS
# ══════════════════════════════════════════════════════════════════════════════

def r_mensuel(taux_annuel_pct: float) -> float:
    return taux_annuel_pct / 100 / 12


def mensualite(capital: float, taux_annuel_pct: float, mois: int) -> float:
    if mois == 0 or capital == 0:
        return 0.0
    if taux_annuel_pct == 0:
        return capital / mois
    r = r_mensuel(taux_annuel_pct)
    return (capital * r * (1 + r) ** mois) / ((1 + r) ** mois - 1)


def amortir(capital: float, taux_annuel_pct: float, mois: int,
            debut: int = 0) -> pd.DataFrame:
    if capital == 0 or mois == 0:
        return pd.DataFrame(columns=["mois", "paiement", "interet",
                                     "capital_rembourse", "capital_restant"])
    rows, restant = [], capital
    pmt = mensualite(capital, taux_annuel_pct, mois)
    r   = r_mensuel(taux_annuel_pct)
    for i in range(mois):
        int_m   = restant * r
        cap_m   = pmt - int_m
        restant = max(0.0, restant - cap_m)
        rows.append({"mois": debut + i + 1, "paiement": pmt,
                     "interet": int_m, "capital_rembourse": cap_m,
                     "capital_restant": restant})
    return pd.DataFrame(rows)


def compute_lissee(capital: float, taux_annuel_pct: float, n_mois: int,
                   charges_sec: np.ndarray) -> tuple:
    """
    Mensualité lissée T (constante) telle que le crédit principal soit
    intégralement remboursé en n_mois.
    Chaque mois i : paiement_principal = T - charges_sec[i]
    Retourne (T, tableau_amortissement_crédit_principal).
    """
    if capital == 0 or n_mois == 0:
        return 0.0, pd.DataFrame(columns=["mois", "paiement", "interet",
                                           "capital_rembourse", "capital_restant"])
    r = r_mensuel(taux_annuel_pct)

    def simuler(T):
        restant, rows = capital, []
        for i in range(n_mois):
            int_m = restant * r
            pmt_p = max(int_m + 0.01, T - charges_sec[i])
            cap_m = pmt_p - int_m
            restant -= cap_m
            rows.append({"mois": i + 1, "paiement": pmt_p, "interet": int_m,
                         "capital_rembourse": cap_m, "capital_restant": max(0.0, restant)})
        return restant, rows

    lo = mensualite(capital, taux_annuel_pct, n_mois)
    hi = lo * 5
    while simuler(hi)[0] > 0:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if simuler(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.001:
            break

    T = (lo + hi) / 2
    _, rows = simuler(T)
    return T, pd.DataFrame(rows)


def eur(v: float) -> str:
    return f"{v:,.0f} \u20ac".replace(",", "\u202f")

def pct(v: float) -> str:
    return f"{v:.2f} %"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏠 Paramètres du projet")
    st.divider()

    st.markdown("### Projet immobilier")
    prix_bien = st.number_input("Prix du bien (€)", min_value=50_000, max_value=1_000_000,
                                value=280_000, step=5_000)
    apport    = st.number_input("Apport personnel (€)", min_value=0, max_value=300_000,
                                value=30_000, step=1_000)

    st.markdown("**Frais de notaire**")
    notaire_pct = st.number_input("Taux frais de notaire (%)", min_value=0.0, max_value=10.0,
                                   value=7.5, step=0.1, format="%.1f",
                                   help="~3 % neuf · ~7-8 % ancien")
    notaire_montant = prix_bien * notaire_pct / 100
    cn1, cn2 = st.columns(2)
    cn1.metric("Frais notaire", eur(notaire_montant))
    cn2.metric("Taux", pct(notaire_pct))

    apport_net       = apport - notaire_montant
    reste_a_financer = prix_bien - max(0.0, apport_net)

    if apport_net < 0:
        st.warning(f"Apport insuffisant pour couvrir les frais de notaire.\nDéficit : {eur(-apport_net)}")
    else:
        st.info(f"**À financer :** {eur(reste_a_financer)}\n\n*Apport net après notaire : {eur(apport_net)}*")

    st.divider()

    st.markdown("### 🟢 PTZ — Prêt à Taux Zéro (0 %)")
    ptz_montant     = st.number_input("Montant PTZ (€)", min_value=0, max_value=150_000,
                                      value=60_000, step=1_000)
    ptz_duree_ans   = st.number_input("Durée totale (ans)", min_value=5, max_value=25,
                                      value=20, step=1, key="ptz_d")
    ptz_differe_ans = st.number_input("Différé — sans remboursement (ans)", min_value=0,
                                      max_value=10, value=5, step=1, key="ptz_diff")
    ptz_duree   = int(ptz_duree_ans) * 12
    ptz_differe = int(ptz_differe_ans) * 12

    st.divider()

    st.markdown("### 🟣 Action Logement — 1 % patronal")
    st.caption("Montant max : 30 000 € · Taux fixe 1 % · Durée max 25 ans")
    credit_un_montant   = st.number_input("Montant (€)", min_value=0, max_value=30_000,
                                          value=20_000, step=500)
    credit_un_duree_ans = st.number_input("Durée (ans)", min_value=1, max_value=25,
                                          value=15, step=1, key="un_d")
    credit_un_duree     = int(credit_un_duree_ans) * 12

    st.divider()

    st.markdown("### 🟡 Crédit principal")
    auto_normal = st.toggle("Montant auto (solde restant)", value=True)
    if auto_normal:
        credit_normal_montant = max(0.0, reste_a_financer - ptz_montant - credit_un_montant)
        st.info(f"**Calculé :** {eur(credit_normal_montant)}")
    else:
        credit_normal_montant = float(st.number_input("Montant (€)", min_value=0,
                                                       max_value=800_000, value=170_000, step=5_000))

    taux_normal             = st.number_input("Taux d'intérêt (%)", min_value=0.5, max_value=7.0,
                                              value=3.5, step=0.05, format="%.2f")
    credit_normal_duree_ans = st.number_input("Durée (ans)", min_value=5, max_value=30,
                                              value=25, step=1, key="norm_d")
    credit_normal_duree     = int(credit_normal_duree_ans) * 12

    st.divider()
    st.caption("Simulateur indicatif — non contractuel.")


# ══════════════════════════════════════════════════════════════════════════════
# CALCULS
# ══════════════════════════════════════════════════════════════════════════════

pmt_ptz    = mensualite(ptz_montant, 0, ptz_duree - ptz_differe)
pmt_un     = mensualite(credit_un_montant, 1, credit_un_duree)
pmt_normal = mensualite(credit_normal_montant, taux_normal, credit_normal_duree)

amo_ptz    = amortir(ptz_montant, 0, ptz_duree - ptz_differe, debut=ptz_differe)
amo_un     = amortir(credit_un_montant, 1, credit_un_duree)
amo_normal = amortir(credit_normal_montant, taux_normal, credit_normal_duree)

total_int_un     = amo_un["interet"].sum()
total_int_normal = amo_normal["interet"].sum()
total_interest   = total_int_un + total_int_normal

# ── Mensualité lissée ─────────────────────────────────────────────────────────
idx_principal   = np.arange(1, credit_normal_duree + 1)
charges_ptz     = np.where((idx_principal > ptz_differe) & (idx_principal <= ptz_duree), pmt_ptz, 0.0)
charges_un      = np.where(idx_principal <= credit_un_duree, pmt_un, 0.0)
charges_sec     = charges_ptz + charges_un

pmt_lissee, amo_lissee = compute_lissee(
    credit_normal_montant, taux_normal, credit_normal_duree, charges_sec
)

total_int_lissee = amo_lissee["interet"].sum() if len(amo_lissee) else 0.0
surcout_lissee   = (total_int_lissee + total_int_un) - total_interest

# ── Timeline mensuelle ────────────────────────────────────────────────────────
max_mois = max(ptz_duree, credit_un_duree, credit_normal_duree, 1)
mois_idx = np.arange(1, max_mois + 1)

def serie(amo, col):
    s = pd.Series(0.0, index=mois_idx)
    if len(amo):
        s.update(amo.set_index("mois")[col])
    return s.values

tl = pd.DataFrame({
    "mois":       mois_idx,
    "ptz":        serie(amo_ptz,    "paiement"),
    "un":         serie(amo_un,     "paiement"),
    "normal":     serie(amo_normal, "paiement"),
    "int_ptz":    serie(amo_ptz,    "interet"),
    "int_un":     serie(amo_un,     "interet"),
    "int_normal": serie(amo_normal, "interet"),
    "cap_ptz":    serie(amo_ptz,    "capital_rembourse"),
    "cap_un":     serie(amo_un,     "capital_rembourse"),
    "cap_normal": serie(amo_normal, "capital_rembourse"),
})
tl["total"]     = tl["ptz"] + tl["un"] + tl["normal"]
tl["total_int"] = tl["int_ptz"] + tl["int_un"] + tl["int_normal"]
tl["cum_int"]   = tl["total_int"].cumsum()
tl["cum_cap"]   = (tl["cap_ptz"] + tl["cap_un"] + tl["cap_normal"]).cumsum()
tl["annee"]     = ((tl["mois"] - 1) // 12) + 1

# Mensualité lissée totale par mois
lissee_principal_paiement = np.zeros(max_mois)
if len(amo_lissee):
    lissee_principal_paiement[:credit_normal_duree] = amo_lissee["paiement"].values
lissee_total = np.zeros(max_mois)
lissee_total[:credit_normal_duree] = lissee_principal_paiement[:credit_normal_duree] + charges_sec

yearly = tl.groupby("annee").agg(
    ptz=("ptz","sum"), un=("un","sum"), normal=("normal","sum"),
    int_total=("total_int","sum"),
).reset_index()

# Sensibilité
rates_arr = np.arange(0.5, 7.05, 0.25)
sens_df = pd.DataFrame({
    "taux": rates_arr,
    "interets": [
        amortir(credit_normal_montant, r, credit_normal_duree)["interet"].sum() + total_int_un
        for r in rates_arr
    ]
})

C = {"ptz":"#4ecdc4","un":"#a78bfa","normal":"#e8c97a",
     "lissee":"#34d399","interest":"#f87171","grid":"#1e2130"}
LAY = dict(paper_bgcolor="#080a10", plot_bgcolor="#0f1117",
           font=dict(color="#c8ccd8", size=12),
           margin=dict(l=10, r=10, t=36, b=20),
           legend=dict(bgcolor="#0f1117", bordercolor="#1e2130", borderwidth=1))


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🏠 Simulateur de Prêts Immobiliers")
st.markdown("PTZ · Action Logement 1 % · Crédit principal · **Mensualité lissée**")
st.divider()

# ── KPIs globaux ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prix du bien",         eur(prix_bien))
c2.metric("Frais de notaire",     eur(notaire_montant), pct(notaire_pct))
c3.metric("Total intérêts",       eur(total_interest),
          f"{total_interest/max(reste_a_financer,1)*100:.1f} % du capital",
          delta_color="inverse")
c4.metric("Coût réel total",
          eur(prix_bien + total_interest + notaire_montant),
          f"dont {eur(total_interest+notaire_montant)} de frais",
          delta_color="inverse")
c5.metric("Durée la plus longue",
          f"{max(ptz_duree_ans,credit_un_duree_ans,credit_normal_duree_ans)} ans")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION MENSUALITÉ LISSÉE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📐 Mensualité lissée")
st.markdown("""
> **Principe :** la banque fixe une mensualité **globale constante** T.  
> Pendant les phases où PTZ et/ou Action Logement sont actifs, le crédit principal rembourse **moins**.  
> Quand ils s'éteignent, le crédit principal rembourse **plus** — mais T reste identique.  
> Résultat : budget parfaitement prévisible, au prix d'un léger surcoût en intérêts.
""")

lk1, lk2, lk3, lk4, lk5 = st.columns(5)
pmt_max_nonlissee = pmt_ptz + pmt_un + pmt_normal
ecart = pmt_lissee - pmt_max_nonlissee

lk1.metric("Mensualité lissée",
           eur(pmt_lissee),
           "Constante sur toute la durée")
lk2.metric("Pic mensualités non-lissées",
           eur(pmt_max_nonlissee),
           "3 prêts simultanément actifs")
lk3.metric("Écart lissée / pic",
           eur(ecart),
           ("Lissée plus élevée" if ecart > 0 else "Lissée plus basse"),
           delta_color="inverse" if ecart > 0 else "normal")
lk4.metric("Surcoût intérêts lissage",
           eur(surcout_lissee),
           "Prix du confort de la constance",
           delta_color="inverse")
lk5.metric("Total intérêts lissée",
           eur(total_int_lissee + total_int_un),
           f"vs {eur(total_interest)} non-lissée",
           delta_color="inverse")

st.markdown("### Mensualité lissée vs mensualités variables")

fig_lissee = go.Figure()

# Zones de phase
phases = [
    (0, ptz_differe, "rgba(78,205,196,0.05)", f"Différé PTZ ({ptz_differe_ans} ans)", C["ptz"]),
    (ptz_differe, min(ptz_duree, credit_un_duree, credit_normal_duree),
     "rgba(167,139,250,0.04)", "", ""),
]
if ptz_differe > 0:
    fig_lissee.add_vrect(x0=0, x1=ptz_differe, fillcolor="rgba(78,205,196,0.06)",
                         layer="below", line_width=0,
                         annotation_text=f"Différé PTZ<br>({ptz_differe_ans} ans)",
                         annotation_position="top left",
                         annotation_font=dict(color=C["ptz"], size=10))

for end_m, label, color in [
    (credit_un_duree, f"Fin AL — M{credit_un_duree}", C["un"]),
    (ptz_duree,       f"Fin PTZ — M{ptz_duree}",      C["ptz"]),
]:
    if end_m < credit_normal_duree:
        fig_lissee.add_vline(x=end_m, line_dash="dot", line_color=color, opacity=0.5,
                             annotation_text=label, annotation_font_color=color,
                             annotation_bgcolor="#0f1117")

# Non-lissée
fig_lissee.add_trace(go.Scatter(
    x=tl["mois"], y=tl["total"],
    name="Non-lissée (variable)", mode="lines",
    line=dict(color=C["interest"], width=2, dash="dot"),
    hovertemplate="Mois %{x}<br>Non-lissée : %{y:,.0f} €<extra></extra>",
))

# Lissée
fig_lissee.add_trace(go.Scatter(
    x=np.arange(1, credit_normal_duree + 1),
    y=lissee_total[:credit_normal_duree],
    name=f"Lissée — {eur(pmt_lissee)}/mois", mode="lines",
    line=dict(color=C["lissee"], width=3),
    fill="tozeroy", fillcolor="rgba(52,211,153,0.07)",
    hovertemplate="Mois %{x}<br>Lissée : %{y:,.0f} €<extra></extra>",
))

fig_lissee.add_hline(y=pmt_lissee, line_dash="dash",
                     line_color=C["lissee"], opacity=0.4,
                     annotation_text=f"  {eur(pmt_lissee)}/mois",
                     annotation_font_color=C["lissee"])

fig_lissee.update_layout(**LAY,
    xaxis=dict(title="Mois", gridcolor=C["grid"]),
    yaxis=dict(title="Mensualité totale (€)", gridcolor=C["grid"]),
    height=400)
st.plotly_chart(fig_lissee, width="stretch")

# ── Décomposition du crédit principal lissé ───────────────────────────────────
if len(amo_lissee):
    with st.expander("🔍 Détail : décomposition intérêts / capital du crédit principal lissé"):
        _m  = amo_lissee["mois"].values
        _i  = amo_lissee["interet"].values
        _cp = amo_lissee["capital_rembourse"].values
        _p  = amo_lissee["paiement"].values

        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=_m, y=_cp, name="Capital remboursé",
                               marker_color=C["lissee"], opacity=0.85))
        fig_d.add_trace(go.Bar(x=_m, y=_i, name="Intérêts",
                               marker_color=C["interest"], opacity=0.85))
        fig_d.add_trace(go.Scatter(x=_m, y=_p,
                                   name="Paiement principal (lissé)",
                                   mode="lines", line=dict(color="#e8c97a", width=2)))
        fig_d.update_layout(**LAY, barmode="stack", height=300,
                            xaxis=dict(title="Mois", gridcolor=C["grid"]),
                            yaxis=dict(title="(€)", gridcolor=C["grid"]))
        st.plotly_chart(fig_d, width="stretch")

    # Cumul intérêts lissée vs non-lissée
    _cum_int_l  = amo_lissee["interet"].cumsum().values
    _m_l        = amo_lissee["mois"].values
    # Non-lissée sur la même durée
    _nl_int     = np.zeros(credit_normal_duree)
    for _, row in amo_normal.iterrows():
        _nl_int[int(row["mois"]) - 1] += row["interet"]
    for _, row in amo_un.iterrows():
        m = int(row["mois"])
        if m <= credit_normal_duree:
            _nl_int[m - 1] += row["interet"]
    _cum_int_nl = np.cumsum(_nl_int)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=np.arange(1, credit_normal_duree + 1), y=_cum_int_nl,
        name="Non-lissée — intérêts cumulés",
        mode="lines", line=dict(color=C["interest"], width=2, dash="dot"),
    ))
    fig_cum.add_trace(go.Scatter(
        x=_m_l, y=_cum_int_l,
        name="Lissée — intérêts cumulés (crédit principal seul)",
        mode="lines", line=dict(color=C["lissee"], width=2.5),
        fill="tonexty", fillcolor="rgba(248,113,113,0.08)",
    ))
    fig_cum.update_layout(**LAY, height=280,
                          xaxis=dict(title="Mois", gridcolor=C["grid"]),
                          yaxis=dict(title="Intérêts cumulés (€)", gridcolor=C["grid"]))
    st.markdown("**Cumul d'intérêts : lissée vs non-lissée**")
    st.plotly_chart(fig_cum, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES STANDARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Analyse des flux")

tab1, tab2, tab3, tab4 = st.tabs([
    "Mensualités annualisées",
    "Cumul capital vs intérêts",
    "Mensualités mensuelles",
    "Sensibilité au taux",
])

with tab1:
    fig_bar = go.Figure()
    for key, label, color in [("ptz","PTZ",C["ptz"]),
                               ("un","Action Logement 1 %",C["un"]),
                               ("normal","Crédit principal",C["normal"])]:
        fig_bar.add_trace(go.Bar(x=yearly["annee"], y=yearly[key],
                                 name=label, marker_color=color, opacity=0.9))
    fig_bar.update_layout(**LAY, barmode="stack", height=400,
                          xaxis=dict(title="Année", tickprefix="An ", gridcolor=C["grid"]),
                          yaxis=dict(title="Remboursements (€)", gridcolor=C["grid"]))
    st.plotly_chart(fig_bar, width="stretch")

with tab2:
    tl3 = tl[tl["mois"] % 3 == 0]
    cross = tl[tl["cum_int"] <= tl["cum_cap"]]
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=tl3["mois"], y=tl3["cum_cap"], mode="lines",
                               name="Capital remboursé",
                               line=dict(color=C["ptz"], width=2.5),
                               fill="tozeroy", fillcolor="rgba(78,205,196,0.10)"))
    fig_a.add_trace(go.Scatter(x=tl3["mois"], y=tl3["cum_int"], mode="lines",
                               name="Intérêts payés",
                               line=dict(color=C["interest"], width=2.5),
                               fill="tozeroy", fillcolor="rgba(248,113,113,0.08)"))
    if len(cross):
        mc = int(cross.iloc[0]["mois"])
        fig_a.add_vline(x=mc, line_dash="dash", line_color="#e8c97a", opacity=0.5,
                        annotation_text=f"Intérêts < Capital — M{mc}",
                        annotation_font_color="#e8c97a", annotation_bgcolor="#0f1117")
    fig_a.update_layout(**LAY, height=400,
                        xaxis=dict(title="Mois", tickprefix="M", gridcolor=C["grid"]),
                        yaxis=dict(title="Cumul (€)", gridcolor=C["grid"]))
    st.plotly_chart(fig_a, width="stretch")

with tab3:
    fig_s = go.Figure()
    for key, label, fc in [
        ("ptz",    "PTZ",                "rgba(78,205,196,0.75)"),
        ("un",     "Action Logement 1 %","rgba(167,139,250,0.75)"),
        ("normal", "Crédit principal",   "rgba(232,201,122,0.75)"),
    ]:
        fig_s.add_trace(go.Scatter(x=tl["mois"], y=tl[key], name=label,
                                   stackgroup="pay", mode="none", fillcolor=fc))
    fig_s.add_trace(go.Scatter(
        x=np.arange(1, credit_normal_duree + 1),
        y=lissee_total[:credit_normal_duree],
        name=f"Lissée ({eur(pmt_lissee)}/mois)", mode="lines",
        line=dict(color=C["lissee"], width=2.5, dash="dash"),
    ))
    if ptz_differe > 0:
        fig_s.add_vrect(x0=0, x1=ptz_differe, fillcolor="rgba(78,205,196,0.04)",
                        layer="below", line_width=0,
                        annotation_text=f"Différé PTZ ({ptz_differe_ans} ans)",
                        annotation_position="top left",
                        annotation_font=dict(color=C["ptz"], size=10))
    fig_s.update_layout(**LAY, height=400,
                        xaxis=dict(title="Mois", gridcolor=C["grid"]),
                        yaxis=dict(title="Mensualité (€)", gridcolor=C["grid"]))
    st.plotly_chart(fig_s, width="stretch")

with tab4:
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["taux"], y=sens_df["interets"],
        mode="lines+markers", name="Total intérêts",
        line=dict(color=C["interest"], width=3),
        marker=dict(color=C["interest"], size=5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.10)",
        hovertemplate="Taux : %{x:.2f} %<br>Intérêts : %{y:,.0f} €<extra></extra>",
    ))
    proche = sens_df.iloc[(sens_df["taux"] - taux_normal).abs().argsort()[:1]]
    fig_sens.add_trace(go.Scatter(
        x=proche["taux"], y=proche["interets"],
        mode="markers+text", name=f"Taux actuel ({pct(taux_normal)})",
        marker=dict(color="#e8c97a", size=13, symbol="diamond"),
        text=[f"  {eur(proche['interets'].values[0])}"],
        textposition="middle right", textfont=dict(color="#e8c97a", size=12),
    ))
    fig_sens.add_vline(x=taux_normal, line_dash="dot", line_color="#e8c97a", opacity=0.4)
    fig_sens.update_layout(**LAY, height=400,
                           xaxis=dict(title="Taux (%)", ticksuffix=" %", gridcolor=C["grid"]),
                           yaxis=dict(title="Total intérêts payés (€)", gridcolor=C["grid"]))
    st.plotly_chart(fig_sens, width="stretch")

st.divider()

# ── Récapitulatif ─────────────────────────────────────────────────────────────
st.markdown("## 📋 Récapitulatif")

recap = pd.DataFrame([
    {"Poste": "Frais de notaire",
     "Montant (€)": notaire_montant,
     "Taux": pct(notaire_pct), "Durée": "—", "Différé": "—",
     "Mensualité non-lissée (€)": 0.0, "Mensualité lissée (€)": 0.0,
     "Total intérêts (€)": 0.0, "Coût total (€)": notaire_montant},

    {"Poste": "PTZ",
     "Montant (€)": ptz_montant,
     "Taux": "0 %", "Durée": f"{ptz_duree_ans} ans",
     "Différé": f"{ptz_differe_ans} ans" if ptz_differe_ans else "—",
     "Mensualité non-lissée (€)": pmt_ptz, "Mensualité lissée (€)": pmt_ptz,
     "Total intérêts (€)": 0.0, "Coût total (€)": ptz_montant},

    {"Poste": "Action Logement 1 %",
     "Montant (€)": credit_un_montant,
     "Taux": "1 %", "Durée": f"{credit_un_duree_ans} ans", "Différé": "—",
     "Mensualité non-lissée (€)": pmt_un, "Mensualité lissée (€)": pmt_un,
     "Total intérêts (€)": total_int_un,
     "Coût total (€)": credit_un_montant + total_int_un},

    {"Poste": "Crédit principal",
     "Montant (€)": credit_normal_montant,
     "Taux": pct(taux_normal), "Durée": f"{credit_normal_duree_ans} ans", "Différé": "—",
     "Mensualité non-lissée (€)": pmt_normal, "Mensualité lissée (€)": pmt_lissee,
     "Total intérêts (€)": total_int_normal,
     "Coût total (€)": credit_normal_montant + total_int_normal},

    {"Poste": "TOTAL",
     "Montant (€)": ptz_montant + credit_un_montant + credit_normal_montant + notaire_montant,
     "Taux": "—",
     "Durée": f"{max(ptz_duree_ans, credit_un_duree_ans, credit_normal_duree_ans)} ans max",
     "Différé": "—",
     "Mensualité non-lissée (€)": pmt_ptz + pmt_un + pmt_normal,
     "Mensualité lissée (€)": pmt_lissee,
     "Total intérêts (€)": total_interest,
     "Coût total (€)": ptz_montant + credit_un_montant + credit_normal_montant + total_interest + notaire_montant},
])

st.dataframe(
    recap.round(0), hide_index=True, width="stretch",
    column_config={
        "Montant (€)":               st.column_config.NumberColumn(format="%d €"),
        "Mensualité non-lissée (€)": st.column_config.NumberColumn(format="%d €"),
        "Mensualité lissée (€)":     st.column_config.NumberColumn(format="%d €"),
        "Total intérêts (€)":        st.column_config.NumberColumn(format="%d €"),
        "Coût total (€)":            st.column_config.NumberColumn(format="%d €"),
    },
)

with st.expander("📥 Tableau d'amortissement complet"):
    frames = [
        amo_ptz.assign(pret="PTZ"),
        amo_un.assign(pret="Action Logement 1 %"),
        amo_normal.assign(pret="Crédit principal (non-lissé)"),
    ]
    if len(amo_lissee):
        frames.append(amo_lissee.assign(pret="Crédit principal (lissé)"))
    amo_full = pd.concat(frames, ignore_index=True)
    amo_full = amo_full.sort_values(["mois", "pret"]).reset_index(drop=True)
    st.dataframe(amo_full.round(2), width="stretch")
    st.download_button("⬇ Télécharger CSV",
                       amo_full.to_csv(index=False).encode("utf-8"),
                       "amortissement.csv", "text/csv")

st.caption("Simulateur indicatif — résultats à titre informatif uniquement, non contractuels.")
