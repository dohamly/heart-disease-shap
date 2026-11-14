"""
app_ui.py
---------
HTML and CSS building blocks of the Streamlit app, kept out of app.py so that the script only
describes the page layout and these pieces can be tested on their own.

Every function returns an HTML string meant for st.markdown(..., unsafe_allow_html=True).
"""

from shap_utils import sigmoid, value_label, variable_label

RED = "#e0546a"    # pushes towards "disease"
GREEN = "#0f9488"  # pushes towards "healthy"

STYLE_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@700;800&display=swap" rel="stylesheet">
<style>
.block-container{padding-top:3.4rem; max-width:1180px}
[data-testid="stSidebar"]{border-right:1px solid #e5e9ee}
[data-testid="stSidebar"] h3{font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:#6b7280; margin-top:.6rem}
.topbar{border-radius:14px; padding:20px 26px; background:linear-gradient(135deg,#0f172a,#134e4a); color:#fff; margin-bottom:12px}
.topbar .tag{font-size:11px; letter-spacing:.06em; color:#7dd3c8; text-transform:uppercase; font-weight:600}
.topbar h1{font-family:'Fraunces',Georgia,serif; font-weight:800; font-size:28px; margin:6px 0 4px; color:#fff; padding:0}
.topbar p{color:#c9d3dc; font-size:13.5px; margin:0}
.disclaimer{background:#fff6e6; border:1px solid #f2d9a3; color:#8a6116; border-radius:10px; padding:10px 14px; font-size:13px; margin-bottom:16px}
.kpi{background:#fff; border:1px solid #e5e9ee; border-radius:12px; padding:14px 16px; height:100%}
.kpi .lbl{font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; font-weight:600}
.kpi .val{font-size:22px; font-weight:800; margin-top:4px; line-height:1.15}
.kpi .sub{font-size:12px; color:#6b7280; margin-top:3px}
.kpi.risk{background:#fdecef; border-color:#f3c5cd} .kpi.risk .val{color:#c23b52}
.kpi.safe{background:#e6f7f5; border-color:#b8e3de} .kpi.safe .val{color:#0b6f66}
.meter{position:relative; height:10px; background:#eceff3; border-radius:6px; margin-top:10px}
.meter .fill{height:100%; border-radius:6px}
.meter .thr{position:absolute; top:-3px; left:50%; width:2px; height:16px; background:#111827}
h4.sec{font-size:15px; margin:18px 0 6px}
.infl{display:grid; grid-template-columns:minmax(0,1.6fr) minmax(0,1fr) 78px; align-items:center; gap:10px; margin-bottom:9px; font-size:13px}
.infl .track{height:14px; background:#f1f3f6; border-radius:5px; overflow:hidden}
.infl .fillbar{height:100%; border-radius:5px}
.infl .v{text-align:right; font-weight:700; font-size:12.5px}
.wf-head,.wf-row{display:grid; grid-template-columns:210px 1fr 52px 58px; align-items:center; gap:10px}
.wf-head{font-size:10.5px; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; margin:12px 0 6px; font-weight:600}
.wf-head span:nth-child(n+3), .wf-x, .wf-p{text-align:right}
.wf-row{margin-bottom:7px; font-size:12.5px}
.wf-row.edge{font-weight:600; padding:5px 0; border-top:1px dashed #e5e9ee}
.wf-row.edge.first{border-top:none; border-bottom:1px dashed #e5e9ee}
.wf-track{position:relative; height:16px; background:#f4f6f8; border-radius:5px}
.wf-bar{position:absolute; top:0; height:100%; border-radius:4px; min-width:3px}
.wf-dot{position:absolute; top:-3px; width:4px; height:22px; background:#111827; border-radius:2px; transform:translateX(-2px)}
.wf-p{color:#6b7280}
.note{background:#e6f7f5; border:1px solid #e5e9ee; border-radius:10px; padding:11px 14px; font-size:12.5px; line-height:1.55; margin-top:12px}
</style>
"""  # noqa: E501 (CSS lines are long by nature)


def risk_color(contribution: float) -> str:
    """Red for a contribution that raises the estimated probability, green otherwise."""
    return RED if contribution > 0 else GREEN


def class_color(predicted_class: int) -> str:
    """Red for the 'disease' class (1), green for the 'healthy' class (0)."""
    return RED if predicted_class == 1 else GREEN


def header_html(disclaimer: str) -> str:
    """Page title banner followed by the educational / non-diagnostic disclaimer."""
    return f"""
<div class="topbar">
  <div class="tag">Prédiction du risque cardiaque · explicable avec SHAP</div>
  <h1>Heart Disease Prediction</h1>
  <p>Saisis un patient dans le panneau de gauche : la prédiction et son explication se mettent à jour automatiquement.</p>
</div>
<div class="disclaimer"><b>Avertissement.</b> {disclaimer}</div>
"""  # noqa: E501


def meter_html(probability: float, color: str) -> str:
    """Horizontal gauge filled up to `probability`, with a tick at the 50 % decision threshold."""
    return (f'<div class="meter"><div class="fill" style="width:{probability * 100:.1f}%;background:{color}"></div>'
            f'<div class="thr"></div></div>')


def kpi_card(label: str, value: str, sub: str = "", css: str = "", value_style: str = "", meter: str = "") -> str:
    """One KPI card: a label, a big value, and optionally a gauge and a small caption."""
    extra_class = f" {css}" if css else ""
    style = f' style="{value_style}"' if value_style else ""
    caption = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi{extra_class}"><div class="lbl">{label}</div>'
            f'<div class="val"{style}>{value}</div>{meter}{caption}</div>')


def influence_rows_html(by_variable, raw: dict, top_n: int = 6) -> str:
    """Bars of the `top_n` clinical variables with the largest SHAP contribution for this patient."""
    top = by_variable.reindex(by_variable.abs().sort_values(ascending=False).index).iloc[:top_n]
    largest = float(top.abs().max()) or 1.0
    rows = ""
    for variable, contribution in top.items():
        color = risk_color(contribution)
        name = f'{variable_label(variable)} <span style="color:#6b7280">({value_label(variable, raw[variable])})</span>'
        width = abs(contribution) / largest * 100
        rows += (f'<div class="infl"><span>{name}</span>'
                 f'<div class="track"><div class="fillbar" style="width:{width:.0f}%;background:{color}"></div></div>'
                 f'<span class="v" style="color:{color}">{contribution:+.2f}</span></div>')
    return rows


def cascade_html(by_variable, raw: dict, base: float, base_probability: float, probability: float,
                 linear: bool, max_steps: int = 8) -> str:
    """
    Waterfall as a table of steps going from the average patient to this patient's prediction.

    The last column gives the cumulative probability after each step. It is only meaningful when
    the SHAP values are in log-odds (`linear`), otherwise it shows a dash.
    """
    ordered = by_variable.reindex(by_variable.abs().sort_values(ascending=False).index)
    shown, others = ordered.iloc[:max_steps], ordered.iloc[max_steps:]
    steps = [(f"{variable_label(v)}: {value_label(v, raw[v])}", float(x)) for v, x in shown.items()]
    if len(others):
        steps.append((f"{len(others)} autres variables", float(others.sum())))

    cumulative, rows = base, []
    for label, effect in steps:
        rows.append((label, effect, cumulative, cumulative + effect))
        cumulative += effect

    visited = [base, cumulative] + [start for _, _, start, _ in rows] + [end for _, _, _, end in rows]
    axis_min, axis_max = min(visited), max(visited)
    padding = (axis_max - axis_min) * 0.06 or 0.1
    axis_min, axis_max = axis_min - padding, axis_max + padding

    def position(value):
        """Horizontal position (in %) of a log-odds value on the shared axis."""
        return (value - axis_min) / (axis_max - axis_min) * 100

    def probability_after(value):
        """Cumulative probability label of a log-odds value."""
        return f"{sigmoid(value) * 100:.0f} %" if linear else "—"

    html = ('<div class="wf-head"><span>Variable : valeur du patient</span><span></span>'
            '<span>Effet</span><span>Prob. cumulée</span></div>'
            '<div class="wf-row edge first"><span>Patient moyen (valeur de base)</span>'
            f'<div class="wf-track"><div class="wf-dot" style="left:{position(base):.2f}%"></div></div><span></span>'
            f'<span class="wf-p"><b>{base_probability * 100:.0f} %</b></span></div>')
    for label, effect, start, end in rows:
        color = risk_color(effect)
        width = abs(end - start) / (axis_max - axis_min) * 100
        html += (f'<div class="wf-row"><span>{label}</span><div class="wf-track">'
                 f'<div class="wf-bar" style="left:{position(min(start, end)):.2f}%;'
                 f'width:{width:.2f}%;background:{color}"></div></div>'
                 f'<span class="wf-x" style="color:{color}">{effect:+.2f}</span>'
                 f'<span class="wf-p">{probability_after(end)}</span></div>')
    html += ('<div class="wf-row edge"><span>Prédiction du modèle</span>'
             f'<div class="wf-track"><div class="wf-dot" style="left:{position(cumulative):.2f}%"></div></div><span></span>'
             f'<span class="wf-p"><b>{probability * 100:.0f} %</b></span></div>')
    return html
