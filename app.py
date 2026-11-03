"""TRANSFORMLAB - Transformation Composition and Inverse Verification System
MAT2303 Linear Algebra & Vector Calculus - Mini Project (Topic 17)
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import engine as E

st.set_page_config(page_title="TRANSFORMLAB", page_icon="🔁", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.pipe {background:#eef2f7; border-radius:10px; padding:14px 18px; font-size:1.05rem;
       border-left:5px solid #2f5d8a; color:#1b2a3a;}
.ok {color:#1a7f37; font-weight:600;} .bad {color:#b42318; font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ presets
V2 = [[1, 0], [0, 1], [1, 1], [2, -1], [3, 2]]
V3 = [[1, 0, 0], [0, 1, 0], [1, 1, 1], [2, -1, 3], [3, 2, 1]]
PRESETS = {
    "Shear → Scale → Rotate 90° (2D)": ([E.shear_x(1), E.scaling(2, 3), E.rotation2d(90)], V2),
    "Rotate 30° → Scale → Reflect (2D)": ([E.rotation2d(30), E.scaling(2, 0.5), E.reflection_x()], V2),
    "Rotate z 90° → Scale → Shear (3D)": ([E.rotation_z(90), E.scaling(2, 1, 3), E.shear_x(1, 3)], V3),
    "Singular case: contains a projection (2D)": ([E.rotation2d(45), E.projection_x(), E.scaling(2, 2)], V2),
    "Custom 2D (identity matrices)": ([E.identity(2)] * 3, V2),
    "Custom 3D (identity matrices)": ([E.identity(3)] * 3, V3),
}

st.title("🔁 TRANSFORMLAB")
st.caption("Transformation Composition and Inverse Verification System · MAT2303 Mini Project · Topic 17")
st.markdown(
    '<div class="pipe"><b>Input vector → T₁ → T₂ → T₃ → Composition matrix C → Reverse transformation C⁻¹ → Verification</b></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Setup")
    name = st.selectbox("Example / preset", list(PRESETS))
    k = st.radio("Number of transformations", [2, 3], index=1, horizontal=True)
    st.markdown("Edit matrices and vectors in the **Inputs** tab. Everything else updates automatically.")
    st.markdown("---")
    st.markdown(
        "**Convention:** column vectors. Applying T₁ first, then T₂, then T₃ gives "
        "T₃(T₂(T₁v)) = (T₃T₂T₁)v, so **C = T₃T₂T₁**."
    )

presetTs, presetV = PRESETS[name]
n = len(presetTs[0])
tabs = st.tabs(["Overview", "Inputs", "Step-by-step", "Composition", "Inverse", "Verification", "Methodology"])

# ------------------------------------------------------------------ inputs
with tabs[1]:
    st.subheader("Transformation matrices")
    st.write("Each Tᵢ is an editable matrix. Change any entry; all results recompute.")
    Ts = []
    cols = st.columns(k)
    labels = [f"x{j+1}" for j in range(n)]
    for i in range(k):
        with cols[i]:
            st.markdown(f"**T{i+1}**")
            df = pd.DataFrame(presetTs[i], columns=labels)
            ed = st.data_editor(df, key=f"T{i}_{name}", hide_index=True, width="stretch")
            Ts.append(ed.fillna(0.0).to_numpy(dtype=float).tolist())
    st.subheader("Input vectors")
    st.write("Each row is one vector. Add or delete rows as needed.")
    vdf = pd.DataFrame(presetV, columns=labels, dtype=float)
    ved = st.data_editor(vdf, key=f"V_{name}", num_rows="dynamic", width="stretch")
    vectors = ved.dropna().to_numpy(dtype=float).tolist()

if not vectors:
    st.warning("Add at least one input vector in the Inputs tab.")
    st.stop()

R = E.analyse(Ts, vectors)
checks = E.verify(Ts, vectors, R)


def mdf(M, rows=None, cols=None):
    d = pd.DataFrame(np.round(np.array(M, dtype=float) + 0.0, 4))
    d.columns = cols or [f"c{j+1}" for j in range(d.shape[1])]
    return d


def vtxt(v):
    return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"


names = ["v"] + ["·".join(f"T{j}" for j in range(i, 0, -1)) + "·v" for i in range(1, k + 1)]
Cname = "·".join(f"T{i}" for i in range(k, 0, -1))

# ------------------------------------------------------------------ overview
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transformations", k)
    c2.metric("Dimension", f"R{n}")
    c3.metric("det(C)", f"{R['det_C']:.4g}")
    c4.metric("Invertible", "Yes" if R["invertible"] else "No")
    st.markdown(f"#### Composition matrix  C = {Cname}")
    st.table(mdf(R["C"], cols=labels))
    if R["invertible"]:
        st.markdown("#### Inverse  C⁻¹")
        st.table(mdf(R["Cinv"], cols=labels))
    else:
        st.error("At least one Tᵢ has determinant 0, so C is singular and C⁻¹ does not exist.")
    passed = sum(1 for c in checks if c[2] is True)
    total = sum(1 for c in checks if c[2] is not None)
    st.success(f"Verification: {passed} of {total} checks passed.")
    st.markdown(
        "**What this tool does:** it applies the transformations one after another to every input vector, "
        "builds the single composition matrix, reverses the process with the inverse matrix, and "
        "verifies every result against an independent calculation."
    )

# ------------------------------------------------------------------ step by step
with tabs[2]:
    st.subheader("Intermediate transformed vectors")
    rows = []
    for i, v in enumerate(vectors):
        st_ = R["stages"][i]
        row = {"Vector": f"v{i+1}"}
        for j, nm in enumerate(names):
            row[nm] = vtxt(st_[j])
        row["C·v (one step)"] = vtxt(R["C_direct"][i])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The last two columns agree: applying T₁, T₂, T₃ one by one gives the same vector as multiplying by C once.")

    if n == 2:
        st.subheader("Geometric view (unit square and the first input vector)")
        sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
        chain_sq = [[list(p) for p in sq]]
        for T in Ts:
            chain_sq.append([E.matvec(T, p) for p in chain_sq[-1]])
        fig, axs = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 3.4))
        ttl = ["Original"] + [f"After T{i+1}" for i in range(k)]
        allpts = np.array([p for s in chain_sq for p in s])
        lim = max(1.5, float(np.abs(allpts).max()) * 1.15)
        v0 = R["stages"][0]
        for j, ax in enumerate(axs):
            ax.add_patch(Polygon(chain_sq[j], closed=True, fc="#9ec5e8", ec="#2f5d8a", alpha=0.7))
            ax.annotate("", xy=v0[j], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
            ax.axhline(0, color="grey", lw=0.6)
            ax.axvline(0, color="grey", lw=0.6)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_title(ttl[j], fontsize=10)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("The geometric plot is shown for 2D transformations. The table above gives the 3D intermediate vectors.")

# ------------------------------------------------------------------ composition
with tabs[3]:
    st.subheader("Composition matrix")
    st.latex(r"C = T_k \cdots T_2 T_1 \qquad\Rightarrow\qquad C\mathbf v = T_k(\cdots T_2(T_1\mathbf v))")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**C = {Cname}**")
        st.table(mdf(R["C"], cols=labels))
        st.write(f"rank(C) = {R['rank_C']},  det(C) = {R['det_C']:.6g}")
    with b:
        st.markdown(f"**Reverse order  {'·'.join(f'T{i}' for i in range(1, k+1))}**")
        st.table(mdf(R["C_rev"], cols=labels))
        diff = E.max_abs_diff(R["C"], R["C_rev"])
        if diff > E.TOL:
            st.warning(f"The two matrices differ (max entry difference {diff:.4g}). Matrix multiplication is not commutative, so the order of transformations matters.")
        else:
            st.info("For these matrices both orders give the same result (they commute).")
    st.subheader("Properties of each transformation")
    pr = pd.DataFrame({"Transformation": [f"T{i+1}" for i in range(k)], "det": [round(d, 6) for d in R["dets"]],
                       "rank": R["ranks"], "Invertible": ["Yes" if abs(d) > E.TOL else "No" for d in R["dets"]]})
    st.dataframe(pr, hide_index=True, width="stretch")
    st.caption("det(C) equals the product of the individual determinants, and C is invertible exactly when every Tᵢ is invertible.")

# ------------------------------------------------------------------ inverse
with tabs[4]:
    st.subheader("Reverse transformation")
    if not R["invertible"]:
        bad = [f"T{i+1}" for i, d in enumerate(R["dets"]) if abs(d) <= E.TOL]
        st.error(f"{', '.join(bad)} has determinant 0 (rank less than {n}), so it collapses some directions and cannot be undone. C⁻¹ does not exist.")
        st.write(f"rank(C) = {R['rank_C']} < {n}: different input vectors can give the same output, so the original cannot be recovered uniquely.")
    else:
        st.latex(r"C^{-1} = T_1^{-1} T_2^{-1} \cdots T_k^{-1}\quad(\text{order reversed})")
        a, b = st.columns(2)
        with a:
            st.markdown("**C⁻¹ by Gauss-Jordan elimination**")
            st.table(mdf(R["Cinv"], cols=labels))
        with b:
            st.markdown("**T₁⁻¹T₂⁻¹…Tₖ⁻¹ (reversal law)**")
            st.table(mdf(R["Cinv_law"], cols=labels))
        st.markdown("**Individual inverses**")
        cs = st.columns(k)
        for i in range(k):
            with cs[i]:
                st.markdown(f"T{i+1}⁻¹")
                st.table(mdf(R["Tinv"][i], cols=labels))
        st.subheader("Undoing the transformations step by step")
        rows = []
        for i, v in enumerate(vectors):
            p = R["back_stages"][i]
            row = {"Vector": f"v{i+1}", "Output C·v": vtxt(p[0])}
            for j in range(1, k + 1):
                row[f"after T{k-j+1}⁻¹"] = vtxt(p[j])
            row["Original v"] = vtxt(v)
            row["Recovered?"] = "✔" if E.vec_diff(p[-1], v) < E.TOL else "✘"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ------------------------------------------------------------------ verification
with tabs[5]:
    st.subheader("Verification report")
    out = []
    for nm, r, ok in checks:
        out.append({"Check": nm, "Max residual": "—" if r is None else f"{r:.2e}",
                    "Result": "—" if ok is None else ("PASS" if ok else "FAIL")})
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
    if R["invertible"]:
        I = E.matmul(R["Cinv"], R["C"])
        st.markdown("**C⁻¹C (should be the identity)**")
        st.table(mdf(I, cols=labels))
    st.caption("All routines (matrix product, determinant, rank, Gauss-Jordan inverse) are written from first principles; NumPy is used only as an independent cross-check.")

# ------------------------------------------------------------------ methodology
with tabs[6]:
    st.subheader("Method")
    st.markdown(
        """
1. **Linear transformation.** A map T is linear if T(au + bv) = aT(u) + bT(v). Every linear map of Rⁿ is multiplication by an n×n matrix.
2. **Algebra of transformations.** The composition (T₂∘T₁)(v) = T₂(T₁(v)) is again linear, and its matrix is the **product** T₂T₁ (note the order).
3. **Composition matrix.** For three maps, C = T₃T₂T₁. Matrix multiplication is associative but not commutative.
4. **Inverse.** T is invertible iff det T ≠ 0 (rank n). The inverse is found by Gauss-Jordan elimination on [T | I].
5. **Reverse transformation.** (T₃T₂T₁)⁻¹ = T₁⁻¹T₂⁻¹T₃⁻¹: the last transformation is undone first.
6. **Verification.** C·v is compared with the step-by-step result; C⁻¹C = CC⁻¹ = I; det(C) = ∏det(Tᵢ); linearity and associativity are tested; results are cross-checked with NumPy.
"""
    )
