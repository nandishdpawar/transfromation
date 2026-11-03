"""TRANSFORMLAB math engine - linear algebra written from first principles.

Convention: column vectors, so applying T1 then T2 then T3 to v gives
T3 (T2 (T1 v)) = (T3 T2 T1) v.  The composition matrix is C = T3 T2 T1.
"""
import math

TOL = 1e-9


# ---------- basic matrix operations ----------
def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(A, B):
    """Matrix product A B:  (AB)_ij = sum_k A_ik B_kj."""
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def matvec(A, v):
    """Matrix-vector product A v."""
    return [sum(A[i][k] * v[k] for k in range(len(v))) for i in range(len(A))]


def transpose(A):
    return [list(r) for r in zip(*A)]


def max_abs_diff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def vec_diff(u, v):
    return max(abs(a - b) for a, b in zip(u, v))


# ---------- elimination based routines ----------
def _eliminate(A):
    """Forward elimination with partial pivoting.
    Returns (echelon matrix, number of row swaps)."""
    M = [list(map(float, r)) for r in A]
    n, m = len(M), len(M[0])
    swaps, r = 0, 0
    for c in range(m):
        if r >= n:
            break
        p = max(range(r, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < TOL:
            continue
        if p != r:
            M[r], M[p] = M[p], M[r]
            swaps += 1
        for i in range(r + 1, n):
            f = M[i][c] / M[r][c]
            for j in range(c, m):
                M[i][j] -= f * M[r][j]
        r += 1
    return M, swaps


def rank(A):
    M, _ = _eliminate(A)
    return sum(1 for row in M if any(abs(x) > TOL for x in row))


def det(A):
    """Determinant by elimination: product of pivots times (-1)^swaps."""
    M, swaps = _eliminate(A)
    d = (-1.0) ** swaps
    for i in range(len(M)):
        d *= M[i][i]
    return d


def inverse(A):
    """Gauss-Jordan elimination on [A | I].  Returns None if A is singular."""
    n = len(A)
    M = [list(map(float, A[i])) + identity(n)[i] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < TOL:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for i in range(n):
            if i != c:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [row[n:] for row in M]


# ---------- standard transformation matrices ----------
def rotation2d(deg):
    t = math.radians(deg)
    c, s = round(math.cos(t), 10), round(math.sin(t), 10)
    return [[c, -s], [s, c]]


def rotation_z(deg):
    t = math.radians(deg)
    c, s = round(math.cos(t), 10), round(math.sin(t), 10)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def scaling(*d):
    n = len(d)
    return [[float(d[i]) if i == j else 0.0 for j in range(n)] for i in range(n)]


def shear_x(k, n=2):
    M = identity(n)
    M[0][1] = float(k)
    return M


def reflection_x():
    return [[1.0, 0.0], [0.0, -1.0]]


def projection_x():
    return [[1.0, 0.0], [0.0, 0.0]]


# ---------- the pipeline ----------
def compose(Ts):
    """Composition matrix C = T_k ... T_2 T_1 for transformations applied in order T_1, T_2, ..."""
    C = Ts[0]
    for T in Ts[1:]:
        C = matmul(T, C)
    return C


def apply_chain(Ts, v):
    """Return [v, T1 v, T2 T1 v, ...] (all intermediate vectors)."""
    out = [list(map(float, v))]
    for T in Ts:
        out.append(matvec(T, out[-1]))
    return out


def analyse(Ts, vectors):
    """Run the whole pipeline and return every intermediate result."""
    k = len(Ts)
    res = {"k": k}
    res["stages"] = [apply_chain(Ts, v) for v in vectors]
    res["C"] = compose(Ts)
    res["C_direct"] = [matvec(res["C"], v) for v in vectors]
    res["dets"] = [det(T) for T in Ts]
    res["ranks"] = [rank(T) for T in Ts]
    res["det_C"] = det(res["C"])
    res["rank_C"] = rank(res["C"])
    res["invertible"] = all(abs(d) > TOL for d in res["dets"])
    # reverse order T1 T2 ... Tk  (to show order matters)
    res["C_rev"] = compose(list(reversed(Ts)))
    if res["invertible"]:
        res["Tinv"] = [inverse(T) for T in Ts]
        res["Cinv"] = inverse(res["C"])
        # reversal law  C^-1 = T1^-1 T2^-1 ... Tk^-1  (= compose of inverses in reversed order)
        res["Cinv_law"] = compose(list(reversed(res["Tinv"])))
        res["recovered"] = [matvec(res["Cinv"], w) for w in res["C_direct"]]
        res["back_stages"] = []
        for i, v in enumerate(vectors):
            w = res["stages"][i][-1]
            path = [w]
            for Ti in reversed(res["Tinv"]):
                path.append(matvec(Ti, path[-1]))
            res["back_stages"].append(path)
    return res


def verify(Ts, vectors, res):
    """List of (check name, residual or None, passed or None)."""
    import numpy as np
    checks = []
    n = len(Ts[0])
    C = res["C"]
    # 1 composition matrix = step by step
    r = max(vec_diff(res["stages"][i][-1], res["C_direct"][i]) for i in range(len(vectors)))
    checks.append(("C v equals T3(T2(T1 v)) for every vector", r, r < TOL))
    # 2 linearity of C
    u, w = vectors[0], vectors[min(1, len(vectors) - 1)]
    a, b = 2.0, -3.0
    lhs = matvec(C, [a * x + b * y for x, y in zip(u, w)])
    rhs = [a * x + b * y for x, y in zip(matvec(C, u), matvec(C, w))]
    r = vec_diff(lhs, rhs)
    checks.append(("Linearity: C(2u - 3w) = 2Cu - 3Cw", r, r < TOL))
    # 3 associativity
    if len(Ts) >= 3:
        left = matmul(matmul(Ts[2], Ts[1]), Ts[0])
        right = matmul(Ts[2], matmul(Ts[1], Ts[0]))
        r = max_abs_diff(left, right)
        checks.append(("Associativity: (T3 T2) T1 = T3 (T2 T1)", r, r < TOL))
    # 4 determinant rule
    prod = 1.0
    for d in res["dets"]:
        prod *= d
    r = abs(res["det_C"] - prod)
    checks.append(("det(C) = det(T1) det(T2) ... det(Tk)", r, r < 1e-8))
    # 5 NumPy cross-check of C
    Cn = np.eye(n)
    for T in Ts:
        Cn = np.array(T, dtype=float) @ Cn
    r = float(np.max(np.abs(Cn - np.array(C))))
    checks.append(("C agrees with NumPy matrix product", r, r < TOL))
    if res["invertible"]:
        Ci = res["Cinv"]
        r = max_abs_diff(matmul(Ci, C), identity(n))
        checks.append(("C^-1 C = I", r, r < TOL))
        r = max_abs_diff(matmul(C, Ci), identity(n))
        checks.append(("C C^-1 = I", r, r < TOL))
        r = max_abs_diff(Ci, res["Cinv_law"])
        checks.append(("Reversal law: C^-1 = T1^-1 T2^-1 ... Tk^-1", r, r < TOL))
        r = max(vec_diff(res["recovered"][i], vectors[i]) for i in range(len(vectors)))
        checks.append(("Reverse transformation recovers every original vector", r, r < TOL))
        r = float(np.max(np.abs(np.linalg.inv(np.array(C)) - np.array(Ci))))
        checks.append(("C^-1 agrees with numpy.linalg.inv", r, r < TOL))
    else:
        checks.append(("Inverse checks skipped: some T_i is singular", None, None))
    return checks
