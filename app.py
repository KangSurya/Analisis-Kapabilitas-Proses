import io
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

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
                                 line=dict(color="#1f77b4")), row=r, col=1)
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
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + icpt, mode="lines", name="Garis normal"))
    fig.update_layout(title=title, height=330, showlegend=False, margin=dict(t=40, b=10),
                      xaxis_title="Kuantil teoritis", yaxis_title="Kuantil sampel")
    return fig


def hist_fig(x, title):
    fig = go.Figure(go.Histogram(x=x, nbinsx=12))
    fig.update_layout(title=title, height=330, margin=dict(t=40, b=10))
    return fig


_cnt = itertools.count()


def show(container, fig, **kw):
    """plotly_chart dengan key unik agar tidak bentrok antar grafik."""
    container.plotly_chart(fig, key=f"pc{next(_cnt)}", **kw)


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


# ------------------------------------------------------------ antarmuka
def render_variable(label, col, bsa, bsb, df):
    use_tf = st.session_state.get(f"tf_{col}", True)
    A = analyze(df, col, bsa, bsb, use_tf)

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
        st.toggle("Terapkan transformasi (hanya untuk Cp dan Cpk)", value=True, key=f"tf_{col}")
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


def main():
    global _cnt
    _cnt = itertools.count()
    st.set_page_config(page_title="Dashboard Kapabilitas Proses", layout="wide")
    st.title("Dashboard Kapabilitas Proses Kopi Arabika Natural Anaerobic")

    with st.sidebar:
        st.header("Input Parameter")
        st.markdown("**Kadar air (%)**")
        bsb_ka = st.number_input("BSB", value=10.0, step=0.1)
        bsa_ka = st.number_input("BSA", value=15.0, step=0.1)
        st.markdown("**Jumlah cacat (gram)**")
        bsa_cc = st.number_input("BSA (spesifikasi satu sisi)", value=60.0, step=1.0)
        berat = st.number_input("Berat sampel uji (gram)", value=300.0, step=10.0)
        st.markdown("**Periode**")
        c1, c2 = st.columns(2)
        d_start = c1.date_input("Mulai", value=pd.to_datetime("2025-10-01"))
        d_end = c2.date_input("Akhir", value=pd.to_datetime("2026-06-30"))
        file = st.file_uploader("Upload data (.xlsx)", type=["xlsx"])
        st.download_button("Download Template", make_template(), "template_data.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        update = st.button("Update", type="primary", width="stretch")

    if update:
        st.session_state["ready"] = False
        try:
            if file is None:
                raise ValueError("Silakan upload file terlebih dahulu.")
            if bsb_ka >= bsa_ka:
                raise ValueError("BSB kadar air harus lebih kecil dari BSA.")
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
            st.session_state.update(ready=True, df=raw, bsb_ka=bsb_ka, bsa_ka=bsa_ka,
                                    bsa_cc=bsa_cc, berat=berat)
        except Exception as e:
            st.error(str(e))

    if not st.session_state.get("ready"):
        st.info("Isi parameter di sidebar, upload data, lalu klik **Update**.")
        return

    S = st.session_state
    df, bsb_ka, bsa_ka, bsa_cc, berat = S["df"], S["bsb_ka"], S["bsa_ka"], S["bsa_cc"], S["berat"]

    t1, t2, t3, t4 = st.tabs(["Kadar Air", "Jumlah Cacat", "Cacat (%)", "Ringkasan & Export"])
    with t1:
        A_ka = render_variable("Kadar Air (%)", "kadar_air", bsa_ka, bsb_ka, df)
    with t2:
        A_cc = render_variable("Jumlah Cacat (gram)", "jumlah_cacat", bsa_cc, None, df)

    with t3:
        ka, cc = df["kadar_air"], df["jumlah_cacat"]
        bawah, atas = int((ka < bsb_ka).sum()), int((ka > bsa_ka).sum())
        cc_gagal = int((cc > bsa_cc).sum())
        ka_ok = (ka >= bsb_ka) & (ka <= bsa_ka)
        cc_ok = cc <= bsa_cc
        st.caption("Dihitung dari seluruh batch pada periode terpilih (termasuk titik yang dihapus dari peta kendali).")
        p1, p2, p3 = st.columns(3)
        show(p1, px.pie(names=["Di bawah BSB", "Di atas BSA", "Sesuai"],
                               values=[bawah, atas, len(df) - bawah - atas], hole=0.35,
                               title="Kadar air",
                               color_discrete_sequence=["#ff7f0e", "#d62728", "#2ca02c"]),
                        width="stretch")
        show(p2, px.pie(names=["Di atas BSA", "Sesuai"], values=[cc_gagal, len(df) - cc_gagal],
                               hole=0.35, title="Jumlah cacat",
                               color_discrete_sequence=["#d62728", "#2ca02c"]), width="stretch")
        show(p3, px.pie(names=["Lolos keduanya", "Gagal kadar air saja", "Gagal cacat saja", "Gagal keduanya"],
                               values=[int((ka_ok & cc_ok).sum()), int((~ka_ok & cc_ok).sum()),
                                       int((ka_ok & ~cc_ok).sum()), int((~ka_ok & ~cc_ok).sum())],
                               hole=0.35, title="Gabungan",
                               color_discrete_sequence=["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]),
                        width="stretch")
        pct = df.assign(cacat_pct=cc / berat * 100)
        fig = px.line(pct, x="no", y="cacat_pct", markers=True,
                      title=f"Persentase cacat per batch (gram / {berat:g} g x 100%)")
        fig.add_hline(y=bsa_cc / berat * 100, line_dash="dash", line_color="red", annotation_text="BSA")
        show(st, fig, width="stretch")
        st.write(f"Rata-rata cacat: **{pct['cacat_pct'].mean():.2f}%** dari berat sampel.")

    with t4:
        rows = [cap_row("Kadar Air", A_ka, bsa_ka, bsb_ka), cap_row("Jumlah Cacat", A_cc, bsa_cc, None)]
        summ = pd.DataFrame(rows)
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


main()
