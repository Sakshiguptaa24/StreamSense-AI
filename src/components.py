"""
Reusable, high-fidelity UI components for StreamSense AI.
Includes a pyarrow-independent HTML/CSS DataFrame renderer to bypass Windows DLL blocks.
"""

import html
import pandas as pd
import streamlit as st

def render_dataframe(df, max_rows: int = 15, height: int = 360, title: str = None):
    """
    Renders a Pandas DataFrame or Series as a high-performance, dark-themed HTML table.
    Completely independent of pyarrow to prevent Application Control DLL block errors.
    """
    if df is None:
        st.markdown(
            '<div class="ss-empty-state">No data available to display.</div>',
            unsafe_allow_html=True,
        )
        return

    # Handle pandas Styler
    if hasattr(df, "data"):
        df = df.data

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if len(df) == 0:
        st.markdown(
            '<div class="ss-empty-state">Dataset contains 0 records.</div>',
            unsafe_allow_html=True,
        )
        return

    total_rows = len(df)
    total_cols = len(df.columns)
    display_rows = min(total_rows, max_rows)
    sub_df = df.head(display_rows)

    # Build Header
    th_cells = []
    th_cells.append('<th class="ss-th ss-th-idx">#</th>')
    for col in sub_df.columns:
        col_name = html.escape(str(col))
        col_type = str(sub_df[col].dtype)
        th_cells.append(
            f'<th class="ss-th">'
            f'<div class="ss-th-content">'
            f'<span class="ss-th-name">{col_name}</span>'
            f'<span class="ss-th-type">{col_type}</span>'
            f'</div>'
            f'</th>'
        )
    thead_html = f"<thead><tr>{''.join(th_cells)}</tr></thead>"

    # Build Rows
    tr_rows = []
    for idx, (_, row) in enumerate(sub_df.iterrows(), start=1):
        td_cells = [f'<td class="ss-td ss-td-idx">{idx}</td>']
        for col in sub_df.columns:
            val = row[col]
            if pd.isna(val):
                cell_content = '<span class="ss-null-pill">null</span>'
            else:
                s_val = str(val)
                # Pretty badges for specific columns
                if col in ["sentiment", "prediction"]:
                    if str(val) in ["1", "positive", "Positive", "1.0"]:
                        cell_content = '<span class="ss-badge ss-badge-pos">● Positive</span>'
                    elif str(val) in ["0", "negative", "Negative", "0.0"]:
                        cell_content = '<span class="ss-badge ss-badge-neg">● Negative</span>'
                    else:
                        cell_content = html.escape(s_val)
                elif col == "Missing %":
                    try:
                        pct = float(val)
                        badge_cls = "ss-badge-neg" if pct > 20 else ("ss-badge-warn" if pct > 5 else "ss-badge-pos")
                        cell_content = f'<span class="ss-badge {badge_cls}">{pct:.1f}%</span>'
                    except Exception:
                        cell_content = html.escape(s_val)
                elif col in ["type", "Content Type"]:
                    if s_val == "Movie":
                        cell_content = '<span class="ss-badge ss-badge-movie">🎬 Movie</span>'
                    elif s_val == "TV Show":
                        cell_content = '<span class="ss-badge ss-badge-show">📡 TV Show</span>'
                    else:
                        cell_content = html.escape(s_val)
                else:
                    if len(s_val) > 280:
                        s_val = s_val[:280] + "…"
                    cell_content = html.escape(s_val)
            td_cells.append(f'<td class="ss-td">{cell_content}</td>')
        tr_rows.append(f'<tr class="ss-tr">{"".join(td_cells)}</tr>')

    tbody_html = f"<tbody>{''.join(tr_rows)}</tbody>"

    title_html = f'<span class="ss-table-title">{html.escape(title)}</span>' if title else ""
    meta_bar = (
        f'<div class="ss-table-meta">'
        f'<span>Showing <strong>{display_rows:,}</strong> of <strong>{total_rows:,}</strong> records ({total_cols} columns)</span>'
        f'{title_html}'
        f'</div>'
    )

    full_html = (
        f'<div class="ss-table-wrapper">'
        f'{meta_bar}'
        f'<div class="ss-table-scroll" style="max-height: {height}px;">'
        f'<table class="ss-table">'
        f'{thead_html}'
        f'{tbody_html}'
        f'</table>'
        f'</div>'
        f'</div>'
    )

    st.markdown(full_html, unsafe_allow_html=True)

def safe_dataframe(data, *args, **kwargs):
    """Drop-in replacement for st.dataframe / st.table that prevents pyarrow DLL loading."""
    render_dataframe(data, max_rows=kwargs.get("max_rows", 15), height=kwargs.get("height", 360))

def kpi_card(label: str, value: str, sub: str = "", color: str = "purple", icon: str = "", delta: str = None):
    """Renders a modern metric card with top accent band."""
    delta_html = f'<div class="ss-kpi-delta">{delta}</div>' if delta else ""
    icon_html = f'<div class="ss-kpi-icon-wrap ss-icon-{color}">{icon}</div>' if icon else ""
    return f"""
    <div class="ss-kpi-card ss-kpi-{color}">
        <div class="ss-kpi-header">
            <span class="ss-kpi-label">{label}</span>
            {icon_html}
        </div>
        <div class="ss-kpi-value">{value}</div>
        <div class="ss-kpi-footer">
            <span class="ss-kpi-sub">{sub}</span>
            {delta_html}
        </div>
    </div>
    """

def page_header(category: str, title: str, desc: str = ""):
    """Renders a top SaaS page header."""
    st.markdown(f"""
    <div class="ss-page-header">
        <div class="ss-breadcrumb">
            <span class="ss-crumb-root">StreamSense AI</span>
            <span class="ss-crumb-sep">›</span>
            <span class="ss-crumb-category">{category}</span>
        </div>
        <h1 class="ss-page-title">{title}</h1>
        {f'<p class="ss-page-desc">{desc}</p>' if desc else ''}
    </div>
    """, unsafe_allow_html=True)

def section_header(title: str, subtitle: str = None):
    """Renders a section separator with subtitle."""
    sub_html = f'<p class="ss-section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="ss-section-wrapper">
        <h3 class="ss-section-title">{title}</h3>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)
