import base64
import io
import itertools
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from streamlit_option_menu import option_menu

# ------------------------------------------------------------ konstanta
D2, D4 = 1.128, 3.267  # n = 2 (D3 = 0)
ALPHA = 0.05
COLS = ["no", "tanggal", "kadar_air", "jumlah_cacat"]
MIN_N = 10


# ------------------------------------------------------------ fungsi inti
def make_template() -> bytes:
    df = pd.DataFrame({
        "no": [1, 2, 3, 4, 5],
        "tanggal": pd.date_range("2025-10-06", periods=5, freq="7D").date,
        "kadar_air": [12.1, 12.8, 11.9, 12.5, 13.0],  # %
        "jumlah_cacat": [30.5, 25.0, 41.2, 28.8, 35.1],  # gram
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="data")
    return buf.getvalue()


def imr(d: pd.DataFrame, col: str) -> pd.DataFrame:
    x = d[col].astype(float).reset_index(drop=True)
    mr = x.diff().abs()
    cl, mrbar = x.mean(), mr.mean()
    out = pd.DataFrame({
        "no": d["no"].values, "tanggal": d["tanggal"].values, "X": x, "MR": mr,
        "CL_I": cl, "UCL_I": cl + 3 * mrbar / D2, "LCL_I": cl - 3 * mrbar / D2,
        "CL_MR": mrbar, "UCL_MR": D4 * mrbar, "LCL_MR": 0.0,
    })
    out["OOC_I"] = (out["X"] > out["UCL_I"]) | (out["X"] < out["LCL_I"])
    out["OOC_MR"] = out["MR"] > out["UCL_MR"]
    return out


def clean_imr(df: pd.DataFrame, col: str, max_iter: int = 20):
    """Hapus titik di luar kendali secara iteratif (MR dulu, lalu I) sampai terkendali."""
    d = df[["no", "tanggal", col]].reset_index(drop=True)
    removed = []
    for it in range(1, max_iter + 1):
        res = imr(d, col)
        if res["OOC_MR"].any():
            idx, chart = res.index[res["OOC_MR"]], "MR"
        elif res["OOC_I"].any():
            idx, chart = res.index[res["OOC_I"]], "I"
        else:
            break
        for i in idx:
            removed.append({"iterasi": it, "peta": chart, "no": int(d.loc[i, "no"]),
                            "tanggal": d.loc[i, "tanggal"], "nilai": float(d.loc[i, col])})
        d = d.drop(idx).reset_index(drop=True)
        if len(d) < MIN_N:
            break
    res = imr(d, col)
    controlled = not (res["OOC_I"].any() or res["OOC_MR"].any())
    return d, res, removed, controlled


def cap_indices(x, bsa, bsb):
    x = np.asarray(x, float)
    sigma = np.abs(np.diff(x)).mean() / D2
    mu = x.mean()
    cpa = (bsa - mu) / (3 * sigma)
    if bsb is None:
        return dict(mu=mu, sigma=sigma, cp=None, cpa=cpa, cpb=None, cpk=cpa)
    cpb = (mu - bsb) / (3 * sigma)
    return dict(mu=mu, sigma=sigma, cp=(bsa - bsb) / (6 * sigma), cpa=cpa, cpb=cpb, cpk=min(cpa, cpb))


def transform(x, bsa, bsb):
    x = np.asarray(x, float)
    if (x > 0).all() and bsa > 0 and (bsb is None or bsb > 0):
        xt, lam = stats.boxcox(x)
        return xt, lam, "Box-Cox", lambda v: stats.boxcox(np.asarray(v, float), lmbda=lam)
    xt, lam = stats.yeojohnson(x)
    return xt, lam, "Yeo-Johnson", lambda v: stats.yeojohnson(np.asarray(v, float), lmbda=lam)


def analyze(df, col, bsa, bsb, use_tf):
    res0 = imr(df[["no", "tanggal", col]], col)
    d, res, removed, controlled = clean_imr(df, col)
    A = dict(col=col, res0=res0, d=d, res=res, removed=removed, controlled=controlled,
             n=len(d), norm=None, tf=None, cap=None, scale=None)
    if not controlled or len(d) < MIN_N:
        return A
    x = d[col].values
    A["norm"] = lilliefors(x, dist="norm")  # (statistik, p-value)
    if A["norm"][1] > ALPHA:
        A["cap"], A["scale"] = cap_indices(x, bsa, bsb), "data asli"
    elif use_tf:
        xt, lam, name, f = transform(x, bsa, bsb)
        s1, p1 = lilliefors(xt, dist="norm")
        tf = dict(name=name, lam=float(lam), xt=xt, stat=s1, p=p1)
        if p1 > ALPHA:
            tf["bsa_t"] = float(f([bsa])[0])
            tf["bsb_t"] = None if bsb is None else float(f([bsb])[0])
            A["cap"] = cap_indices(xt, tf["bsa_t"], tf["bsb_t"])
            A["scale"] = f"data transformasi {name} (λ = {lam:.3f})"
        A["tf"] = tf
    return A


# ------------------------------------------------------------ grafik
def plot_imr(res, label):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=(f"Peta MR - {label}", f"Peta Individu (I) - {label}"))
    rows = [("MR", "CL_MR", "UCL_MR", "LCL_MR", "OOC_MR"), ("X", "CL_I", "UCL_I", "LCL_I", "OOC_I")]
    for r, (v, cl, ucl, lcl, ooc) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=res["no"], y=res[v], mode="lines+markers",
                                 line=dict(color="#00A6E8")), row=r, col=1)
        bad = res[res[ooc]]
        fig.add_trace(go.Scatter(x=bad["no"], y=bad[v], mode="markers",
                                 marker=dict(color="red", size=11, symbol="x")), row=r, col=1)
        for name, color, dash in [(cl, "green", "solid"), (ucl, "red", "dash"), (lcl, "red", "dash")]:
            fig.add_trace(go.Scatter(x=res["no"], y=res[name], mode="lines",
                                     line=dict(color=color, dash=dash)), row=r, col=1)
    fig.update_layout(height=600, showlegend=False, margin=dict(t=40, b=10))
    fig.update_xaxes(title_text="Produksi ke-", row=2, col=1)
    return fig


def qq_fig(x, title):
    (osm, osr), (slope, icpt, _) = stats.probplot(x, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data", marker=dict(color="#00A6E8")))
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + icpt, mode="lines", name="Garis normal"))
    fig.update_layout(title=title, height=330, showlegend=False, margin=dict(t=40, b=10),
                      xaxis_title="Kuantil teoritis", yaxis_title="Kuantil sampel")
    return fig


def hist_fig(x, title):
    fig = go.Figure(go.Histogram(x=x, nbinsx=12, marker_color="#0090D0"))
    fig.update_layout(title=title, height=330, margin=dict(t=40, b=10))
    return fig


_cnt = itertools.count()
GRID = "rgba(255,255,255,0.10)"


def style_fig(fig):
    """Tema gelap transparan agar menyatu dengan latar dashboard."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                      title=dict(font=dict(size=16), x=0))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show(container, fig, **kw):
    """plotly_chart dengan key unik agar tidak bentrok antar grafik."""
    container.plotly_chart(style_fig(fig), key=f"pc{next(_cnt)}", **kw)


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def cap_row(name, A, bsa, bsb):
    row = dict(Variabel=name, n_dipakai=A["n"], Titik_dihapus=len(A["removed"]),
               BSB=bsb, BSA=bsa, Terkendali=A["controlled"])
    if A["norm"]:
        row["p_normal_asli"] = A["norm"][1]
    if A["tf"]:
        row.update(Transformasi=A["tf"]["name"], Lambda=A["tf"]["lam"], p_normal_transformasi=A["tf"]["p"])
    if A["cap"]:
        c = A["cap"]
        row.update(Skala=A["scale"], Cp=c["cp"], CpA=c["cpa"], CpB=c["cpb"], Cpk=c["cpk"],
                   Kapabel=("Ya" if (c["cp"] if c["cp"] is not None else c["cpk"]) > 1 else "Tidak"))
    return row



# ------------------------------------------------------------ halaman variabel
def render_variable(label, col, bsa, bsb, A):

    st.markdown("#### 1. Peta kendali MR dan I")
    if A["removed"]:
        st.warning(f"{len(A['removed'])} titik di luar kendali dihapus sementara dari perhitungan. "
                   "Penyebabnya (assignable causes) tetap perlu ditelusuri.")
        st.dataframe(pd.DataFrame(A["removed"]), width="stretch", hide_index=True)
    show(st, plot_imr(A["res"], label), width="stretch")
    r = A["res"].iloc[0]
    st.caption(f"I: BKA {r.UCL_I:.3f} | GT {r.CL_I:.3f} | BKB {r.LCL_I:.3f}   ||   "
               f"MR: BKA {r.UCL_MR:.3f} | GT {r.CL_MR:.3f} | BKB 0")
    with st.expander("Peta kendali awal (sebelum penghapusan titik)"):
        show(st, plot_imr(A["res0"], label), width="stretch")

    if not A["controlled"] or A["n"] < MIN_N:
        st.error("Proses belum terkendali atau data tersisa terlalu sedikit "
                 f"(< {MIN_N}). Analisis kapabilitas tidak dapat dilakukan.")
        return A
    if A["n"] < 20:
        st.warning(f"Data tersisa hanya {A['n']} observasi, hasil kurang andal.")

    st.markdown("#### 2. Uji normalitas (Kolmogorov-Smirnov/Lilliefors)")
    stat0, p0 = A["norm"]
    x = A["d"][col].values
    st.write(f"Data asli: D = {stat0:.4f}, p-value = {p0:.4f} "
             f"({'normal' if p0 > ALPHA else 'tidak normal'}, α = {ALPHA})")
    if p0 <= ALPHA:
        v = st.toggle("Terapkan transformasi (hanya untuk Cp dan Cpk)",
                      value=st.session_state.get(f"tfp_{col}", True), key=f"tf_{col}")
        st.session_state[f"tfp_{col}"] = v
    c1, c2 = st.columns(2)
    show(c1, hist_fig(x, "Histogram data asli"), width="stretch")
    show(c2, qq_fig(x, "Q-Q plot data asli"), width="stretch")

    tf = A["tf"]
    if tf:
        st.write(f"Transformasi **{tf['name']}**, λ = {tf['lam']:.4f}. Setelah transformasi: "
                 f"D = {tf['stat']:.4f}, p-value = {tf['p']:.4f} "
                 f"({'normal' if tf['p'] > ALPHA else 'masih tidak normal'}).")
        c3, c4 = st.columns(2)
        show(c3, hist_fig(tf["xt"], "Histogram setelah transformasi"), width="stretch")
        show(c4, qq_fig(tf["xt"], "Q-Q plot setelah transformasi"), width="stretch")
        if "bsa_t" in tf:
            st.caption(f"Batas spesifikasi setelah transformasi: BSA = {tf['bsa_t']:.4f}"
                       + (f", BSB = {tf['bsb_t']:.4f}" if tf["bsb_t"] is not None else ""))

    st.markdown("#### 3. Indeks kapabilitas proses")
    c = A["cap"]
    if c is None:
        st.error("Data tidak berdistribusi normal, sehingga Cp dan Cpk tidak valid. "
                 "Aktifkan transformasi atau gunakan pendekatan distribusi non-normal.")
        return A
    st.caption(f"Dihitung dari {A['scale']}; σ̂ = MR̄/d₂ = {c['sigma']:.4f}")
    m = st.columns(4)
    m[0].metric("Cp (presisi)", fmt(c["cp"]))
    m[1].metric("Cpk (akurasi)", fmt(c["cpk"]))
    m[2].metric("CpA", fmt(c["cpa"]))
    m[3].metric("CpB", fmt(c["cpb"]))
    key = c["cp"] if c["cp"] is not None else c["cpk"]
    if bsb is None:
        st.info("Spesifikasi satu sisi: hanya CpA/Cpk yang dihitung, Cp tidak didefinisikan.")
    (st.success if key > 1 else st.error)(
        "Proses KAPABEL (indeks > 1)" if key > 1 else "Proses TIDAK KAPABEL (indeks ≤ 1)")
    return A

# ------------------------------------------------------------ tampilan
CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] {background-color:#00152A;}
[data-testid="stHeader"] {background:rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color:#003B73;}
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {background-color:#00152A !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {background-color:#0A2A4D;}
.side-sub {text-align:center;color:#cfe0f3;font-size:.9rem;margin:-4px 0 22px 0;}
.side-title {font-size:1.5rem;font-weight:700;margin:6px 0 4px 0;color:#fff;}
.kpi-head {background:#0B3A6B;border-radius:4px;padding:14px 16px;font-size:1.05rem;color:#fff;}
.kpi-cap {margin-top:18px;font-size:.9rem;color:#dbe7f3;}
.kpi-val {font-size:2.3rem;font-weight:600;color:#fff;line-height:1.25;}
.chart-title {font-weight:700;font-size:1.05rem;margin:28px 0 -6px 8px;color:#fff;}
h1, h2, h3, h4 {color:#fff;}
hr {border-color:rgba(255,255,255,0.15);}
</style>
"""

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 250">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#F0245F"/><stop offset="1" stop-color="#2F6BFF"/></linearGradient></defs>
<polygon points="150,8 236,56 236,150 150,198 64,150 64,56" fill="none" stroke="url(#g)" stroke-width="16"/>
<polygon points="150,58 196,84 196,128 150,154 104,128 104,84" fill="none" stroke="url(#g)" stroke-width="12"/>
<text x="150" y="236" text-anchor="middle" font-family="Arial" font-weight="800" font-size="22"
 textLength="280" lengthAdjust="spacingAndGlyphs"><tspan fill="#E91E63">KHATULISTIWA </tspan><tspan fill="#2F8BFF">COFFEE</tspan></text>
</svg>"""


def sidebar_logo():
    if os.path.exists("logo.png"):  # ganti dengan logo perusahaan jika ada
        st.image("logo.png", width="stretch")
    else:
        b64 = base64.b64encode(LOGO_SVG.encode()).decode()
        st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;padding:0 10px;">',
                    unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Online Analytics</div>', unsafe_allow_html=True)


def kpi(container, label, caption, value):
    container.markdown(f'<div class="kpi-head">📌 {label}</div><div class="kpi-cap">{caption}</div>'
                       f'<div class="kpi-val">{value}</div>', unsafe_allow_html=True)


def chart_title(container, text):
    container.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------ halaman
def page_home(df, A_ka, A_cc, bsb_ka, bsa_ka, bsb_cc, bsa_cc):
    view = st.selectbox("Tampilan", ["Grafik", "Tabular"], label_visibility="collapsed")
    c = st.columns(5)
    cap_ka, cap_cc = A_ka["cap"], A_cc["cap"]
    kpi(c[0], "Jumlah Produksi", "batch", len(df))
    kpi(c[1], "Rata-rata Kadar Air", "mean %", f"{df['kadar_air'].mean():.2f}")
    kpi(c[2], "Rata-rata Cacat", "mean gram", f"{df['jumlah_cacat'].mean():.1f}")
    kpi(c[3], "Kadar Air", "Cp / Cpk",
        f"{cap_ka['cp']:.2f} / {cap_ka['cpk']:.2f}" if cap_ka else "N/A")
    kpi(c[4], "Jumlah Cacat", "Cp / Cpk",
        f"{cap_cc['cp']:.2f} / {cap_cc['cpk']:.2f}" if cap_cc else "N/A")
    st.markdown("---")

    if view == "Tabular":
        st.dataframe(df, width="stretch", hide_index=True)
        return

    ka, cc = df["kadar_air"], df["jumlah_cacat"]
    ka_ok = (ka >= bsb_ka) & (ka <= bsa_ka)
    cc_ok = (cc >= bsb_cc) & (cc <= bsa_cc)
    l, r = st.columns(2)
    chart_title(l, "Kadar Air per Produksi (%)")
    fig = go.Figure(go.Scatter(x=df["no"], y=ka, mode="lines+markers", line=dict(color="#00A6E8")))
    for v, t in [(bsb_ka, "BSB"), (bsa_ka, "BSA")]:
        fig.add_hline(y=v, line_dash="dash", line_color="red", annotation_text=t)
    fig.update_layout(height=380, margin=dict(t=20), xaxis_title="Produksi ke-", yaxis_title="Kadar air (%)")
    show(l, fig, width="stretch")

    chart_title(r, "Status Batch terhadap Spesifikasi")
    status = pd.DataFrame({
        "Status": ["Lolos keduanya", "Gagal kadar air saja", "Gagal cacat saja", "Gagal keduanya"],
        "Jumlah": [int((ka_ok & cc_ok).sum()), int((~ka_ok & cc_ok).sum()),
                   int((ka_ok & ~cc_ok).sum()), int((~ka_ok & ~cc_ok).sum())]})
    fig2 = px.bar(status.iloc[::-1], x="Jumlah", y="Status", orientation="h",
                  color_discrete_sequence=["#0090D0"])
    fig2.update_layout(height=380, margin=dict(t=20))
    show(r, fig2, width="stretch")


def page_cacat(df, bsb_ka, bsa_ka, bsb_cc, bsa_cc, berat):
    ka, cc = df["kadar_air"], df["jumlah_cacat"]
    bawah, atas = int((ka < bsb_ka).sum()), int((ka > bsa_ka).sum())
    cc_bawah, cc_atas = int((cc < bsb_cc).sum()), int((cc > bsa_cc).sum())
    ka_ok = (ka >= bsb_ka) & (ka <= bsa_ka)
    cc_ok = (cc >= bsb_cc) & (cc <= bsa_cc)
    st.caption("Dihitung dari seluruh batch pada periode terpilih (termasuk titik yang dihapus dari peta kendali).")
    p1, p2, p3 = st.columns(3)
    show(p1, px.pie(names=["Di bawah BSB", "Di atas BSA", "Sesuai"],
                    values=[bawah, atas, len(df) - bawah - atas], hole=0.35, title="Kadar air",
                    color_discrete_sequence=["#ff7f0e", "#d62728", "#00A6E8"]), width="stretch")
    show(p2, px.pie(names=["Di bawah BSB", "Di atas BSA", "Sesuai"],
                    values=[cc_bawah, cc_atas, len(df) - cc_bawah - cc_atas],
                    hole=0.35, title="Jumlah cacat",
                    color_discrete_sequence=["#ff7f0e", "#d62728", "#00A6E8"]), width="stretch")
    show(p3, px.pie(names=["Lolos keduanya", "Gagal kadar air saja", "Gagal cacat saja", "Gagal keduanya"],
                    values=[int((ka_ok & cc_ok).sum()), int((~ka_ok & cc_ok).sum()),
                            int((ka_ok & ~cc_ok).sum()), int((~ka_ok & ~cc_ok).sum())],
                    hole=0.35, title="Gabungan",
                    color_discrete_sequence=["#00A6E8", "#ff7f0e", "#9467bd", "#d62728"]), width="stretch")
    pct = df.assign(cacat_pct=cc / berat * 100)
    fig = px.line(pct, x="no", y="cacat_pct", markers=True,
                  title=f"Persentase cacat per batch (gram / {berat:g} g x 100%)",
                  color_discrete_sequence=["#00A6E8"])
    fig.add_hline(y=bsa_cc / berat * 100, line_dash="dash", line_color="red", annotation_text="BSA")
    if bsb_cc > 0:
        fig.add_hline(y=bsb_cc / berat * 100, line_dash="dash", line_color="orange", annotation_text="BSB")
    show(st, fig, width="stretch")
    st.write(f"Rata-rata cacat: **{pct['cacat_pct'].mean():.2f}%** dari berat sampel.")


def page_export(df, A_ka, A_cc, bsb_ka, bsa_ka, bsb_cc, bsa_cc):
    summ = pd.DataFrame([cap_row("Kadar Air", A_ka, bsa_ka, bsb_ka),
                         cap_row("Jumlah Cacat", A_cc, bsa_cc, bsb_cc)])
    st.dataframe(summ, width="stretch", hide_index=True)
    removed = pd.DataFrame([dict(variabel=n, **r) for n, A in
                            [("Kadar Air", A_ka), ("Jumlah Cacat", A_cc)] for r in A["removed"]])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        A_ka["res"].to_excel(w, sheet_name="IMR_Kadar_Air", index=False)
        A_cc["res"].to_excel(w, sheet_name="IMR_Jumlah_Cacat", index=False)
        removed.to_excel(w, sheet_name="Titik_Dihapus", index=False)
        summ.to_excel(w, sheet_name="Kapabilitas", index=False)
        df.to_excel(w, sheet_name="Data_Asli", index=False)
    st.download_button("Export hasil ke Excel", buf.getvalue(), "hasil_analisis.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ------------------------------------------------------------ main
def load_data(file, bsb_ka, bsa_ka, bsb_cc, bsa_cc, d_start, d_end):
    if file is None:
        raise ValueError("Silakan upload file terlebih dahulu.")
    if bsb_ka >= bsa_ka:
        raise ValueError("BSB kadar air harus lebih kecil dari BSA.")
    if bsb_cc >= bsa_cc:
        raise ValueError("BSB jumlah cacat harus lebih kecil dari BSA.")
    if d_start > d_end:
        raise ValueError("Tanggal mulai harus sebelum tanggal akhir.")
    raw = pd.read_excel(file)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    miss = [c for c in COLS if c not in raw.columns]
    if miss:
        raise ValueError(f"Kolom tidak ditemukan: {miss}. Gunakan template.")
    raw = raw[COLS].copy()
    raw["tanggal"] = pd.to_datetime(raw["tanggal"], errors="coerce")
    for c in ["no", "kadar_air", "jumlah_cacat"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna()
    if ((raw["kadar_air"] < 0) | (raw["kadar_air"] > 100) | (raw["jumlah_cacat"] < 0)).any():
        raise ValueError("Terdapat nilai tidak valid (kadar air di luar 0-100 atau cacat negatif).")
    raw = raw[(raw["tanggal"].dt.date >= d_start) & (raw["tanggal"].dt.date <= d_end)]
    raw = raw.sort_values("tanggal").reset_index(drop=True)
    if len(raw) < MIN_N:
        raise ValueError(f"Data pada periode terpilih kurang dari {MIN_N} observasi.")
    return raw


def main():
    global _cnt
    _cnt = itertools.count()
    st.set_page_config(page_title="Dashboard", page_icon="☕", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        sidebar_logo()
        st.markdown('<div class="side-title">Input Parameter</div>', unsafe_allow_html=True)
        bsb_ka = st.number_input("BSB Kadar Air (%)", value=10.0, step=0.1)
        bsa_ka = st.number_input("BSA Kadar Air (%)", value=15.0, step=0.1)
        bsb_cc = st.number_input("BSB Jumlah Cacat (gram)", value=0.0, step=1.0)
        bsa_cc = st.number_input("BSA Jumlah Cacat (gram)", value=60.0, step=1.0)
        berat = st.number_input("Berat sampel uji (gram)", value=300.0, step=10.0)
        c1, c2 = st.columns(2)
        d_start = c1.date_input("Mulai", value=pd.to_datetime("2025-10-01"))
        d_end = c2.date_input("Akhir", value=pd.to_datetime("2026-06-30"))
        file = st.file_uploader("Upload data (.xlsx)", type=["xlsx"])
        st.download_button("Download Template", make_template(), "template_data.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")
        update = st.button("Update", type="primary", width="stretch")
        st.write("")
        page = option_menu(
            "Main Menu", ["Home", "Kadar Air", "Jumlah Cacat", "Cacat (%)", "Export"],
            icons=["house", "droplet", "box-seam", "pie-chart", "download"],
            menu_icon="display", default_index=0,
            styles={"container": {"padding": "12px", "background-color": "#0A2A4D", "border-radius": "10px"},
                    "menu-title": {"color": "white", "font-size": "24px"},
                    "icon": {"color": "white", "font-size": "18px"},
                    "nav-link": {"color": "white", "font-size": "17px", "text-align": "left",
                                 "margin": "3px 0", "--hover-color": "#17508F"},
                    "nav-link-selected": {"background-color": "#1B4F8A"}})

    st.title("☕ Dashboard Kapabilitas Proses Kopi Arabika")

    if update:
        st.session_state["ready"] = False
        try:
            raw = load_data(file, bsb_ka, bsa_ka, bsb_cc, bsa_cc, d_start, d_end)
            st.session_state.update(ready=True, df=raw, bsb_ka=bsb_ka, bsa_ka=bsa_ka,
                                    bsb_cc=bsb_cc, bsa_cc=bsa_cc, berat=berat)
        except Exception as e:
            st.error(str(e))

    if not st.session_state.get("ready"):
        st.info("Isi parameter di sidebar, upload data, lalu klik **Update**.")
        return

    S = st.session_state
    df, bsb_ka, bsa_ka, berat = S["df"], S["bsb_ka"], S["bsa_ka"], S["berat"]
    bsb_cc, bsa_cc = S.get("bsb_cc", 0.0), S["bsa_cc"]
    A_ka = analyze(df, "kadar_air", bsa_ka, bsb_ka, S.get("tfp_kadar_air", True))
    A_cc = analyze(df, "jumlah_cacat", bsa_cc, bsb_cc, S.get("tfp_jumlah_cacat", True))

    st.header(f"Page: {page}")
    if page == "Home":
        page_home(df, A_ka, A_cc, bsb_ka, bsa_ka, bsb_cc, bsa_cc)
    elif page == "Kadar Air":
        render_variable("Kadar Air (%)", "kadar_air", bsa_ka, bsb_ka, A_ka)
    elif page == "Jumlah Cacat":
        render_variable("Jumlah Cacat (gram)", "jumlah_cacat", bsa_cc, bsb_cc, A_cc)
    elif page == "Cacat (%)":
        page_cacat(df, bsb_ka, bsa_ka, bsb_cc, bsa_cc, berat)
    else:
        page_export(df, A_ka, A_cc, bsb_ka, bsa_ka, bsb_cc, bsa_cc)


main()
