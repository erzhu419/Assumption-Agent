# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `Causally Coupled Multi-Center Panel Deconfounding, Interferometric Demodulation & Connected Box-L0 Certification` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_sparse_cert_and_combinatorial_pruning` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `383,460` (`366,348` / `94,336` / `17,112`) |
| **Total Episode Wall Time (s)** | `651.12s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14275 ms` LLM)
- **Rationale**: Read the verifier before designing the certificate: its bound formulas and tolerances determine which relaxation and branching strategy can certify the solution. Inspect only compact study metadata.

```bash
ls -lah /holdout_task; sed -n '1,300p' /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np,scipy
x=pickle.load(open('/holdout_task/public_eq02_studies.pkl','rb')); print(type(x)); print(list(x) if isinstance(x,dict) else len(x));
for s in (x.values() if isinstance(x,dict) else x):
 if isinstance(s,dict):
  for k,v in s.items(): print(k, list(v) if isinstance(v,dict) else (np.shape(v),v if np.ndim(v)==0 else ''))
PY
```
- **Container Exit Code**: `1` (`632 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq02_studies.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`17592 ms` LLM)
- **Rationale**: A bounded search can distinguish relocated public fixtures from a missing mount; that determines whether the next step can use the actual verifier and studies or must implement against the stated contract and test synthetic cases.
- **Alternatives Weighed**: `Assume the documented public files are available` (rejected: The first command established that /holdout_task does not exist in this container; search the advertised pretrain location before substituting synthetic verification.)

```bash
pwd; ls -la /; ls -la /pretrain_task; find /pretrain_task /app /workspace /mnt /opt -maxdepth 4 -type f \( -iname '*eq02*' -o -iname '*schema*' -o -iname '*interferometr*' -o -name 'AGENTS.md' \) -print 2>/dev/null; python3 -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/pretrain_task
total 60
drwxr-xr-x   1 root root 4096 Nov 26 04:50 .
drwxr-xr-x   1 root root 4096 Nov 26 04:50 ..
-rwxr-xr-x   1 root root    0 Nov 26 04:50 .dockerenv
lrwxrwxrwx   1 root root    7 Apr 22  2024 bin -> usr/bin
drwxr-xr-x   2 root root 4096 Apr 22  2024 boot
drwxr-xr-x   5 root root  340 Nov 26 04:50 dev
drwxr-xr-x   1 root root 4096 Nov 26 04:50 etc
drwxr-xr-x   3 root root 4096 Nov 10 14:56 home
lrwxrwxrwx   1 root root    7 Apr 22  2024 lib -> usr/lib
lrwxrwxrwx   1 root root    9 Apr 22  2024 lib64 -> usr/lib64
drwxr-xr-x   2 root root 4096 Nov 10 14:48 media
drwxr-xr-x   2 root root 4096 Nov 10 14:48 mnt
drwxr-xr-x   2 root root 4096 Nov 10 14:48 opt
drwxr-xr-x   2 root root 4096 Nov 23 09:31 pretrain_task
dr-xr-xr-x 652 root root    0 Nov 26 04:50 proc
drwx------   2 root root 4096 Nov 10 14:55 root
drwxr-xr-x   4 root root 4096 Nov 10 14:56 run
lrwxrwxrwx   1 root root    8 Apr 22  2024 sbin -> usr/sbin
drwxr-xr-x   2 root root 4096 Nov 10 14:48 srv
dr-xr-xr-x  13 root root    0 Nov 26 04:50 sys
drwxrwxrwt   2 root root 4096 Nov 10 14:56 tmp
drwxr-xr-x  12 root root 4096 Nov 10 14:48 usr
drwxr-xr-x  11 root root 4096 Nov 10 14:55 var
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 04:50 ..
1.26.4 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`98806 ms` LLM)
- **Rationale**: The diagonal Gram minorant yields separable exact coordinate bounds on each subcube. Compile the first implementation to catch syntax failures before checking certificate validity, tightness, and runtime on independently generated connected studies.
- **Alternatives Weighed**: `Enumerate every support` (rejected: Enumeration does not by itself yield the contracted certificate with at most 2048 leaves.)

```bash
mkdir -p /holdout_task && cat > /holdout_task/coupled_interferometry_certified_l0.py <<'PY'
"""Reference-panel calibration and certified symmetric-box L0 selection."""

import heapq

import numpy as np
from scipy.optimize import minimize


def _align(panel):
    x = np.array(panel['X_panel_raw'], dtype=np.float64, copy=True)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    centers = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    cell_size = np.empty(len(x), dtype=np.float64)
    for center in np.unique(centers):
        mask = centers == center
        c = controls[mask]
        valid = np.isfinite(c)
        cols = valid.sum(axis=0) >= 2
        c = np.where(valid[:, cols], c[:, cols], np.nan)
        offset = np.median(np.nanmean(c, axis=0))
        gain = np.median(np.nanstd(c, axis=0, ddof=0))
        x[mask] = (x[mask] - offset) / gain
        for stratum in (0, 1, 2):
            cell = mask & (strata == stratum)
            cell_size[cell] = cell.sum()
    weight = 1 / cell_size
    weight /= weight.mean()
    finite = np.isfinite(x)
    means = (weight @ np.where(finite, x, 0)) / (weight @ finite)
    return np.where(finite, x, means)


def _calibrate(h, y):
    beta = np.linalg.lstsq(h, y, rcond=None)[0]
    # An L1 start keeps large calibration glitches from setting the Cauchy scale.
    for _ in range(35):
        r = y - h @ beta
        sw = (r * r + 1e-6) ** -0.25
        nxt = np.linalg.lstsq(h * sw[:, None], y * sw, rcond=None)[0]
        if np.linalg.norm(nxt - beta) <= 1e-8 * (1 + np.linalg.norm(beta)):
            beta = nxt
            break
        beta = nxt
    r = y - h @ beta
    scale = max(1.5 * 1.4826 * np.median(np.abs(r - np.median(r))), 1e-7)
    for _ in range(100):
        sw = 1 / np.hypot(1, (y - h @ beta) / scale)
        nxt = np.linalg.lstsq(h * sw[:, None], y * sw, rcond=None)[0]
        if np.linalg.norm(nxt - beta) <= 1e-10 * (1 + np.linalg.norm(beta)):
            return nxt
        beta = nxt
    return beta


def _diagonal(g):
    # Begin with a scale-invariant eigenvalue bound, then use the remaining
    # PSD budget to increase individual coordinates via rank-one downdates.
    norm = np.sqrt(np.maximum(g.diagonal(), np.finfo(float).tiny))
    scaled = g / norm[:, None] / norm
    smallest = np.linalg.eigvalsh(scaled)[0]
    if smallest <= 1e-10:
        return np.zeros(len(g))
    lo = 0.45 * smallest
    r = scaled - lo * np.eye(len(g))
    inv = np.linalg.inv(r)
    d = np.full(len(g), lo)
    for _ in range(4):
        for j in np.argsort(-1 / inv.diagonal()):
            v = inv[:, j].copy()
            inc = 0.38 / inv[j, j]
            d[j] += inc
            inv += (inc / (1 - inc * v[j])) * np.outer(v, v)
    d *= norm**2
    # Strict slack for the certificate's PSD check.
    return 0.999999 * d


class _Selector:
    def __init__(self, x, y, lam2, lam0, box):
        self.x, self.y = x, y
        self.lam2, self.lam0, self.box = lam2, lam0, box
        self.p = x.shape[1]
        self.g = x.T @ x
        self.b = x.T @ y
        self.const = 0.5 * (y @ y)
        self.d = _diagonal(self.g)
        self.q = self.g - np.diag(self.d)
        self.a = self.d + lam2
        self.ridge = self.g + lam2 * np.eye(self.p)
        self.slope = np.full(self.p, lam0 / box)
        self.knee = np.full(self.p, box)
        positive = self.a > 0
        self.knee[positive] = np.minimum(box, np.sqrt(2 * lam0 / self.a[positive]))
        if lam0 > 0:
            self.slope = lam0 / self.knee + 0.5 * self.a * self.knee
        else:
            self.slope.fill(0)
        self.best = self.const
        self.w = np.zeros(self.p)
        self.seen = set()
        self.tol = 5e-6
        self.pad = min(1e-7, 32 * np.finfo(float).eps * (1 + self.const + box * np.linalg.norm(self.b, 1) + box**2 * np.sum(np.abs(self.g))))

    def _objective(self, w):
        r = self.x @ w - self.y
        return float(0.5 * (r @ r + self.lam2 * (w @ w)) + self.lam0 * np.count_nonzero(np.abs(w) > 1e-8))

    def _fit_support(self, active):
        key = tuple(np.flatnonzero(active))
        if not key or key in self.seen:
            return
        self.seen.add(key)
        jj = np.array(key)
        a = self.ridge[np.ix_(jj, jj)]
        b = self.b[jj]
        try:
            t = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            t = np.linalg.lstsq(a, b, rcond=None)[0]
        box = self.box
        if np.any(np.abs(t) > box):
            def loss(v):
                av = a @ v
                return 0.5 * v @ av - b @ v, av - b
            t = minimize(loss, np.clip(t, -box, box), method='L-BFGS-B', jac=True,
                         bounds=[(-box, box)] * len(t),
                         options={'ftol': 1e-15, 'gtol': 2e-11, 'maxiter': 240}).x
            # Polish the free coordinates after identifying saturated ones.
            free = (np.abs(t) < box - 1e-9) | (t * (a @ t - b) > 0)
            if free.any():
                try:
                    u = np.linalg.solve(a[np.ix_(free, free)], b[free] - a[np.ix_(free, ~free)] @ t[~free])
                    if np.all(np.abs(u) <= box):
                        t[free] = u
                except np.linalg.LinAlgError:
                    pass
        w = np.zeros(self.p)
        w[jj] = t
        w[np.abs(w) <= 1e-8] = 0
        value = self._objective(w)
        if value < self.best:
            self.best, self.w = value, w

    def _bound(self, c, state):
        qc = self.q @ c
        lin = qc - self.b
        t = np.clip(np.divide(-lin, self.a, out=-np.sign(lin) * self.box,
                              where=self.a > 0), -self.box, self.box)
        v = self.lam0 + t * (lin + 0.5 * self.a * t)
        v[state == 0] = 0
        free = state == -1
        v[free] = np.minimum(v[free], 0)
        return float(self.const - 0.5 * (c @ qc) + v.sum() - self.pad)

    def _node(self, state, start):
        free = state == -1
        active = state == 1
        allowed = np.flatnonzero(state != 0)
        c = start.copy()
        c[state == 0] = 0
        q, a, b, box = self.q, self.a, self.b, self.box
        slope, knee = self.slope, self.knee
        denom = q.diagonal() + a
        qdiag = q.diagonal()
        qcols = np.ascontiguousarray(q.T)
        residual = b - q @ c
        best_lb = self._bound(c, state)
        best_c = c.copy()
        sweeps = 130 if len(allowed) == self.p else 85
        for sweep in range(sweeps):
            largest = 0.0
            for j in allowed:
                old = c[j]
                rhs = residual[j] + qdiag[j] * old
                if active[j]:
                    new = rhs / denom[j] if denom[j] > 0 else np.sign(rhs) * box
                else:
                    mag = abs(rhs)
                    if mag <= slope[j]:
                        new = 0.0
                    elif mag <= qdiag[j] * knee[j] + slope[j] and qdiag[j] > 0:
                        new = np.copysign((mag - slope[j]) / qdiag[j], rhs)
                    else:
                        new = rhs / denom[j] if denom[j] > 0 else np.sign(rhs) * box
                new = min(box, max(-box, new))
                delta = new - old
                if delta:
                    c[j] = new
                    residual -= delta * qcols[j]
                    largest = max(largest, abs(delta))
            if sweep % 5 == 4 or largest < 2e-10 or sweep + 1 == sweeps:
                lb = self._bound(c, state)
                if lb > best_lb:
                    best_lb, best_c = lb, c.copy()
                if best_lb >= self.best - self.tol or largest < 2e-10:
                    break
        if best_lb < self.best - self.tol:
            frac = np.divide(np.abs(c), knee, out=np.zeros(self.p), where=knee > 0)
            for threshold in (0.5, 0.12, 0.88):
                self._fit_support(active | (free & (frac >= threshold)))
            if not free.any() and best_lb < self.best - self.tol:
                # The exact support optimizer also supplies the tangent point.
                if np.array_equal(np.abs(self.w) > 1e-8, active):
                    lb = self._bound(self.w, state)
                    if lb > best_lb:
                        best_lb, best_c = lb, self.w.copy()
        else:
            frac = np.zeros(self.p)
        return best_lb, best_c, frac

    def _leaf(self, state, lb, center):
        return {'fixed_zero': (state == 0).tolist(), 'fixed_active': (state == 1).tolist(),
                'lb': lb, 'minorant_diagonal': self.d.tolist(), 'minorant_center': center.tolist()}

    def solve(self):
        root = np.full(self.p, -1, dtype=np.int8)
        lb, center, frac = self._node(root, self.w)
        # Improve the initial incumbent by true one-flip box-ridge refits.
        for _ in range(4):
            previous = self.best
            support = np.abs(self.w) > 1e-8
            for j in range(self.p):
                neighbor = support.copy()
                neighbor[j] = not neighbor[j]
                self._fit_support(neighbor)
            if self.best >= previous - 1e-10:
                break
        heap = [(lb, 0, root, center, frac)]
        leaves = []
        serial = 0
        while heap:
            lb, _, state, center, frac = heapq.heappop(heap)
            free = state == -1
            if lb >= self.best - self.tol:
                leaves.append(self._leaf(state, lb, center))
                continue
            if not free.any():
                raise RuntimeError('Optimality gap remains on a fully fixed support')
            score = np.where(free, np.minimum(frac, 1 - frac), -np.inf)
            j = int(np.argmax(score))
            if score[j] <= 0:
                j = int(np.argmax(np.where(free, np.abs(center), -np.inf)))
            for value in (int(frac[j] >= 0.5), int(frac[j] < 0.5)):
                child = state.copy()
                child[j] = value
                child_lb, child_c, child_frac = self._node(child, center)
                if child_lb >= self.best - self.tol:
                    leaves.append(self._leaf(child, child_lb, child_c))
                else:
                    serial += 1
                    heapq.heappush(heap, (child_lb, serial, child, child_c, child_frac))
        if len(leaves) > 2048:
            raise RuntimeError(f'Certificate requires {len(leaves)} leaves')
        lower = min(leaf['lb'] for leaf in leaves)
        return {'weights': self.w.tolist(), 'objective': self.best,
                'certified_lower_bound': min(lower, self.best), 'subcube_leaves': leaves}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    """Align the panel, remove calibration effects, and certify feature selection."""
    panel = study['panel']
    strata = np.asarray(panel['stratum'])
    aligned = _align(panel)
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    ref = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(ref))
    h = np.column_stack((np.ones(len(strata)), strata == 1, strata == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    beta = _calibrate(h[cal], y[cal])
    deconfounded = y[target] - h[target] @ beta
    selection = _Selector(np.asarray(study['X_cand'], dtype=np.float64), deconfounded,
                          float(study['lam2']), float(study['lam0']), float(study['M'])).solve()
    return {'aligned_panel': aligned, 'calibration_weights': beta,
            'deconfounded_target': deconfounded, 'certified_selection': selection}
PY
python3 -I -m py_compile /holdout_task/coupled_interferometry_certified_l0.py
```
- **Container Exit Code**: `0` (`420 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0e24947d36bdd3c2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0e24947d (directory).
```

### Turn 4 — `execute_in_remote_docker` (`58846 ms` LLM)
- **Rationale**: Independently reconstruct the panel and causal handoff, compare a small box problem to exhaustive face enumeration, and recompute leaf bounds and partition coverage on connected 18- and 19-column cases. Failures distinguish preprocessing errors from certificate or runtime bottlenecks.

```bash
cat > /tmp/test_eq02.py <<'PY'
import importlib.util
import itertools
import os
import time
import numpy as np

path = '/holdout_task/coupled_interferometry_certified_l0.py'
spec = importlib.util.spec_from_file_location('eq02', path)
eq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eq)

def check(x,y,l2,l0,m,out):
    p=x.shape[1]; w=np.asarray(out['weights']); r=x@w-y
    value=.5*(r@r+l2*(w@w))+l0*np.count_nonzero(abs(w)>1e-8)
    assert w.shape==(p,) and np.max(abs(w))<=m+1e-12 and abs(value-out['objective'])<1e-8
    leaves=out['subcube_leaves']; assert 1<=len(leaves)<=2048
    universe=np.arange(1<<p,dtype=np.int32); covered=np.zeros(len(universe),dtype=np.int16)
    g=x.T@x; b=x.T@y; base=.5*(y@y); smallest=np.inf; slack=np.inf
    for leaf in leaves:
        z,a=[leaf[k] for k in ('fixed_zero','fixed_active')]
        assert all(len(v)==p and all(type(q) is bool for q in v) for v in (z,a))
        z,a=np.asarray(z),np.asarray(a); assert not (z&a).any()
        ab=sum(1<<int(j) for j in np.flatnonzero(a)); zb=sum(1<<int(j) for j in np.flatnonzero(z))
        covered[(universe&(ab|zb))==ab]+=1
        d,c=[np.asarray(leaf[k]) for k in ('minorant_diagonal','minorant_center')]
        assert d.shape==c.shape==(p,) and np.all(d>=0)
        q=g-np.diag(d); slack=min(slack,np.linalg.eigvalsh(q)[0]); assert slack>=-1e-9
        coef=q@c-b; curv=d+l2
        t=np.array([np.clip(-u/v,-m,m) if v>0 else -np.sign(u)*m for u,v in zip(coef,curv)])
        f=l0+coef*t+.5*curv*t*t
        recomputed=base-.5*c@(q@c)+sum(f[a])+sum(np.minimum(f[~(a|z)],0))
        assert leaf['lb']<=recomputed+1e-8
        smallest=min(smallest,leaf['lb'])
    assert np.all(covered==1)
    gap=value-out['certified_lower_bound']; assert 0<=gap<=1e-5 and abs(out['certified_lower_bound']-smallest)<1e-8
    return len(leaves),gap,slack

def design(rng,n,p,rho):
    x=rng.normal(size=(n,p))
    if rho:
        x=np.sqrt(1-rho)*x+np.sqrt(rho)*rng.normal(size=(n,1))*rng.choice([-1,1],size=p)
    return x/np.linalg.norm(x,axis=0)*rng.uniform(1.7,3.0,p)

def panel_trial(seed,p,rho):
    rng=np.random.default_rng(seed); n=980; ns=210
    probs=np.geomspace(1,7,15); probs/=sum(probs)
    cells=np.r_[np.arange(15),rng.choice(15,n-15,p=probs)]; rng.shuffle(cells)
    c,s=cells//3,cells%3
    c_names=np.array([17,29,8,41,94])[c]
    gain=np.array([.52,.85,1.1,1.6,2.0])[c,None]; offset=np.array([-1.4,.0,1.3,1.8,-.8])[c,None]
    x=rng.normal(size=(n,8))+s[:,None]*rng.normal(0,.27,(1,8)); raw=x*gain+offset
    controls=rng.normal(size=(n,18))*gain+offset
    raw[rng.random(raw.shape)<.025]=np.nan; controls[rng.random(controls.shape)<.05]=np.nan
    controls[c==0,0]=np.nan; controls[np.flatnonzero(c==0)[0],0]=1000
    freq=np.bincount(cells); wt=1/freq[cells]; expected=raw.copy()
    for k in range(5):
        cc=controls[c==k]; cols=[v[np.isfinite(v)] for v in cc.T if np.isfinite(v).sum()>=2]
        expected[c==k]=(raw[c==k]-np.median([v.mean() for v in cols]))/np.median([v.std() for v in cols])
    for j in range(8):
        ok=np.isfinite(expected[:,j]); expected[~ok,j]=np.average(expected[ok,j],weights=wt[ok])
    phase=rng.uniform(-1.5,1.5,(n,2)); carrier=rng.uniform(-np.pi,np.pi,(n,2)); ref=1.35*np.exp(1j*carrier); sig=.88*np.exp(1j*(carrier+phase))
    beta=rng.uniform(-.9,.9,13); h=np.column_stack([np.ones(n),s==1,s==2,x,phase]); target=rng.choice(n,ns,replace=False); cal=np.setdiff1d(np.arange(n),target)
    y=h@beta+rng.normal(0,.11,n); glitches=rng.choice(cal,round(.12*len(cal)),replace=False); y[glitches]+=np.clip(4*rng.standard_cauchy(len(glitches)),-11,11)
    xx=design(rng,ns,p,rho); signal=np.zeros(p); idx=rng.choice(p,5,replace=False); signal[idx]=rng.uniform(.60,1.55,len(idx))*rng.choice([-1,1],len(idx)); y[target]+=xx@signal
    study=dict(panel=dict(center_id=c_names,stratum=s,X_panel_raw=raw,controls=controls),interferometry=dict(signal_phasor=sig,ref_phasor=ref), y_assay_raw=y,calibration_indices=cal,stage3_indices=target,X_cand=xx,lam2=.08,lam0=.28,M=1.1)
    start=time.perf_counter(); out=eq.solve_coupled_interferometry_l0(study); elapsed=time.perf_counter()-start
    aligned=out['aligned_panel']; theta=out['calibration_weights']; dy=out['deconfounded_target']
    np.testing.assert_allclose(aligned,expected,atol=2e-14)
    np.testing.assert_allclose(dy,y[target]-np.column_stack([np.ones(n),s==1,s==2,aligned,phase])[target]@theta,atol=2e-14)
    mae=np.mean(abs(aligned-x)); rel=np.linalg.norm(theta-beta)/np.linalg.norm(beta)
    summary=check(xx,dy,.08,.28,1.1,out['certified_selection']); print('panel',seed,p,rho,'MAE',round(mae,6),'relL2',round(rel,6),'leaves/gap/psd',summary,'seconds',round(elapsed,4),flush=True)
    assert mae<=.11 and rel<=.14
    return elapsed

def small_oracle():
    rng=np.random.default_rng(42); x=design(rng,24,6,.71); y=rng.normal(size=24)*1.6; l2=.1; l0=.21; m=.35
    output=eq._Selector(x,y,l2,l0,m).solve(); ref=np.inf
    g=x.T@x+l2*np.eye(6); b=x.T@y
    for face in itertools.product((0,1,2,3),repeat=6):
        f=np.array(face); w=np.zeros(6); w[f==2]=-m; w[f==3]=m; free=f==1
        if free.any():
            w[free]=np.linalg.solve(g[np.ix_(free,free)], b[free]-g[np.ix_(free,~free)]@w[~free])
        if np.max(abs(w))>m: continue
        r=x@w-y; ref=min(ref,.5*(r@r+l2*w@w)+l0*np.count_nonzero(abs(w)>1e-8))
    assert abs(ref-output['objective'])<=1e-5
    print('face oracle error',abs(ref-output['objective']),'leaves/gap/psd',check(x,y,l2,l0,m,output),flush=True)

if __name__=='__main__':
    if os.path.isfile('/holdout_task/public_eq02_studies.pkl'):
        print('Public fixtures now present',flush=True)
    small_oracle()
    elapsed=[panel_trial(k,p,r) for k,p,r in [(91,18,.34),(92,19,.72),(93,19,.86),(96,18,.92)]]
    assert sum(sorted(elapsed)[-2:])<=18
PY
python3 -I /tmp/test_eq02.py
```
- **Container Exit Code**: `0` (`6872 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
face oracle error 0.0 leaves/gap/psd (3, 3.979039320256561e-13, 4.294711511082793e-05)
panel 91 18 0.34 MAE 0.040617 relL2 0.016898 leaves/gap/psd (2, 2.5481359244849955e-09, 2.686373436008174e-06) seconds 1.3482
panel 92 19 0.72 MAE 0.047785 relL2 0.014953 leaves/gap/psd (3, 2.5343859633153443e-07, 9.92604307488153e-07) seconds 1.0746
panel 93 19 0.86 MAE 0.045117 relL2 0.013148 leaves/gap/psd (4, 1.6866508190105378e-11, 5.37788130857206e-07) seconds 1.4819
panel 96 18 0.92 MAE 0.04613 relL2 0.015413 leaves/gap/psd (5, 1.48947520983711e-11, 2.036738236890544e-07) seconds 1.4823

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0e24947d36bdd3c2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0e24947d (directory).
```

### Turn 5 — `execute_in_remote_docker` (`23588 ms` LLM)
- **Rationale**: The end-to-end tests used sparse, well-separated signals. Near-threshold supports and nearly redundant columns test whether the fixed-support optimizer, branch count, or runtime become limiting. A failure would determine which part of the selector to improve.

```bash
cat > /tmp/stress_eq02.py <<'PY'
from test_eq02 import eq, check, design
import time
import numpy as np

for seed,p,kind in [(17,19,'near-threshold'),(18,18,'near-threshold'),(29,19,'paired'),(33,18,'paired'),(51,19,'cancellation')]:
    rng=np.random.default_rng(seed); n=120; lam0=.35; lam2=.035; m=.52
    if kind=='near-threshold':
        x=design(rng,n,p,.30)
        g=x.T@x
        z=rng.choice([-1.,1.],p)*np.sqrt(2*lam0*g.diagonal())*rng.uniform(.88,1.12,p)
        y=x@np.linalg.solve(g,z)+.07*rng.normal(size=n)
    elif kind=='paired':
        a=design(rng,n,(p+1)//2,.10)
        x=np.repeat(a,2,axis=1)[:,:p]+.005*rng.normal(size=(n,p))
        y=a@(rng.choice([-1.,1.],a.shape[1])*rng.uniform(.47,.59,a.shape[1]))+.04*rng.normal(size=n)
    else:
        x=design(rng,n,p,.997)
        w=np.zeros(p); w[:8]=rng.choice([-1.,1.],8)*rng.uniform(1.1,1.4,8)
        y=x@w+.055*rng.normal(size=n)
    t=time.perf_counter()
    try:
        ans=eq._Selector(x,y,lam2,lam0,m).solve()
        elapsed=time.perf_counter()-t
        print(seed,p,kind,'seconds',round(elapsed,4),'certificate',check(x,y,lam2,lam0,m,ans),flush=True)
    except RuntimeError as e:
        print(seed,p,kind,'seconds',round(time.perf_counter()-t,4),'FAIL',str(e),flush=True)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -c 'import sys; sys.path.insert(0,"/tmp"); import stress_eq02'
```
- **Container Exit Code**: `0` (`12398 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
17 19 near-threshold seconds 0.0248 certificate (8, 1.7497114868092467e-12, 1.5110531430117875e-06)
18 18 near-threshold seconds 0.134 certificate (32, 5.452438500697099e-11, 1.5581613850355493e-06)
29 19 paired seconds 5.5863 certificate (2043, 4.1870062972293454e-11, 1.6309460541448242e-09)
33 18 paired seconds 3.9636 certificate (1977, 4.500829430931219e-06, 1.5540403777169285e-09)
51 19 cancellation seconds 0.0937 certificate (36, 6.2940763712049375e-12, 6.7420068657020095e-09)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0e24947d36bdd3c2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0e24947d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`35381 ms` LLM)
- **Rationale**: The calibration and objective checks have comfortable margins, but the paired-column certificate does not. Two nearby designs probe whether the leaf limit is exceeded, while the existing pair under default settings checks whether the prior single-thread runtime result transfers. Either failure identifies what to fix.
- **Alternatives Weighed**: `Change the branching rule immediately` (rejected: The current method passed but came within five leaves of the cap; test nearby connected designs and default-runtime behavior to distinguish a leaf-count bottleneck from a runtime bottleneck before editing.)

```bash
python3 -I - <<'PY'
import sys, time
sys.path.insert(0, '/tmp')
from test_eq02 import eq, check, design
import numpy as np
for seed,p in [(29,19),(33,18),(31,19),(34,18)]:
    rng=np.random.default_rng(seed); n=120
    a=design(rng,n,(p+1)//2,.10)
    x=np.repeat(a,2,axis=1)[:,:p]+.005*rng.normal(size=(n,p))
    y=a@(rng.choice([-1.,1.],a.shape[1])*rng.uniform(.47,.59,a.shape[1]))+.04*rng.normal(size=n)
    t=time.perf_counter()
    try:
        out=eq._Selector(x,y,.035,.35,.52).solve()
        elapsed=time.perf_counter()-t
        print(seed,p,'seconds',round(elapsed,3),'certificate',check(x,y,.035,.35,.52,out),flush=True)
    except RuntimeError as e:
        print(seed,p,'seconds',round(time.perf_counter()-t,3),'FAIL',e,flush=True)
PY
```
- **Container Exit Code**: `0` (`47195 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
29 19 seconds 17.147 certificate (2043, 4.1869618883083604e-11, 1.630945902658355e-09)
33 18 seconds 8.071 certificate (1977, 4.5008294295989515e-06, 1.5540405724113469e-09)
31 19 seconds 13.919 certificate (1202, 9.577967663432219e-08, 1.7461407470365332e-09)
34 18 seconds 4.296 certificate (1480, 3.977250406794752e-06, 1.600584409587609e-09)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0e24947d36bdd3c2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0e24947d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`38283 ms` LLM)
- **Rationale**: Use small positive-definite solves and reuse each fully fixed support fit as its tangent center, even when that support is not the incumbent. Include the objective’s penalty-free threshold band in the leaf bound. Retest under default settings to see whether runtime still requires a branching change and whether the tighter objective accounting preserves the certificate gap.
- **Alternatives Weighed**: `Require single-thread BLAS settings from the grader` (rejected: The runtime contract must hold without changing the caller’s environment.)

```bash
cp /holdout_task/coupled_interferometry_certified_l0.py /tmp/eq02_v01.py
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/coupled_interferometry_certified_l0.py')
s=p.read_text().replace('from scipy.optimize import minimize','from scipy.linalg import solve as linear_solve\nfrom scipy.optimize import minimize')
s=s.replace('self.seen = set()', 'self.fits = {}')
s=s.replace('        if not key or key in self.seen:\n            return\n        self.seen.add(key)', '        if not key:\n            return np.zeros(self.p)\n        if key in self.fits:\n            return self.fits[key]')
s=s.replace('np.linalg.solve(a, b)', "linear_solve(a, b, assume_a='pos', check_finite=False)")
s=s.replace('np.linalg.solve(a[np.ix_(free, free)], b[free] - a[np.ix_(free, ~free)] @ t[~free])', "linear_solve(a[np.ix_(free, free)], b[free] - a[np.ix_(free, ~free)] @ t[~free],\n                                     assume_a='pos', check_finite=False)")
s=s.replace('        w[np.abs(w) <= 1e-8] = 0\n        value', '        self.fits[key] = w\n        value').replace('            self.best, self.w = value, w\n','            self.best, self.w = value, w\n        return w\n')
s=s.replace('        v[state == 0] = 0\n        free = state == -1\n        v[free] = np.minimum(v[free], 0)', '        # An inactive coefficient pays no penalty anywhere in the threshold band.\n        tiny = min(self.box, 1e-8)\n        z = np.clip(t, -tiny, tiny)\n        zero_cost = z * (lin + 0.5 * self.a * z)\n        fixed_zero = state == 0\n        v[fixed_zero] = zero_cost[fixed_zero]\n        free = state == -1\n        v[free] = np.minimum(v[free], zero_cost[free])')
s=s.replace('        c[state == 0] = 0\n        q, a, b, box', '        c[state == 0] = 0\n        if not free.any():\n            w = self._fit_support(active)\n            return self._bound(w, state), w, np.zeros(self.p)\n        q, a, b, box')
a=s.index('            if not free.any() and best_lb < self.best - self.tol:')
b=s.index('        else:\n            frac',a)
s=s[:a]+s[b:]
p.write_text(s)
PY
python3 -I /tmp/test_eq02.py && python3 -I -c 'import sys; sys.path.insert(0,"/tmp"); import stress_eq02'
```
- **Container Exit Code**: `0` (`16521 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
face oracle error 0.0 leaves/gap/psd (3, 3.540662518730642e-08, 4.294711511082793e-05)
panel 91 18 0.34 MAE 0.040617 relL2 0.016898 leaves/gap/psd (2, 6.563260068759291e-08, 2.686373436008174e-06) seconds 1.2525
panel 92 19 0.72 MAE 0.047785 relL2 0.014953 leaves/gap/psd (3, 2.9283006952596224e-07, 9.92604307488153e-07) seconds 0.4923
panel 93 19 0.86 MAE 0.045117 relL2 0.013148 leaves/gap/psd (4, 3.932719749855096e-08, 5.37788130857206e-07) seconds 0.5765
panel 96 18 0.92 MAE 0.04613 relL2 0.015413 leaves/gap/psd (5, 4.456537538999328e-08, 2.036738236890544e-07) seconds 0.6955
17 19 near-threshold seconds 0.2092 certificate (8, 2.300915902253564e-08, 1.5110531429215819e-06)
18 18 near-threshold seconds 0.2836 certificate (32, 7.206165975048862e-08, 1.5581613862151613e-06)
29 19 paired seconds 5.5527 certificate (2043, 4.243869256015387e-09, 1.630945902658355e-09)
33 18 paired seconds 3.7436 certificate (1977, 4.576760213481634e-06, 1.5540405724113469e-09)
51 19 cancellation seconds 0.2053 certificate (36, 1.8632329457091146e-08, 6.742006863692847e-09)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=82714184b77073fd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_82714184 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`24830 ms` LLM)
- **Rationale**: Retest the two paired cases not yet run on this candidate and add two fresh seeds. A leaf-cap failure would justify changing branching or compaction; passing would support keeping the simpler implementation. If the public files have appeared, use the official validator before submitting.
- **Alternatives Weighed**: `Submit the current candidate immediately` (rejected: Runtime has improved, but the tightest paired-column certificate still has only five leaves of headroom.)

```bash
python3 -I - <<'PY'
import sys, time
from pathlib import Path
sys.path.insert(0, '/tmp')
from test_eq02 import eq, check, design
import numpy as np
root = Path('/holdout_task')
public = root / 'public_eq02_studies.pkl'
validator = root / 'validate_public_schema.py'
print('public studies:', public.is_file(), '| public validator:', validator.is_file(), flush=True)
for seed, p in ((31,19),(34,18),(35,19),(36,19)):
    rng=np.random.default_rng(seed)
    a=design(rng,120,(p+1)//2,.10)
    x=np.repeat(a,2,axis=1)[:,:p]+.005*rng.normal(size=(120,p))
    y=a@(rng.choice([-1.,1.],a.shape[1])*rng.uniform(.47,.59,a.shape[1]))+.04*rng.normal(size=120)
    t=time.perf_counter()
    try:
        out=eq._Selector(x,y,.035,.35,.52).solve()
    except RuntimeError as exc:
        print(seed,p,'FAIL',str(exc),round(time.perf_counter()-t,3),'s',flush=True)
        continue
    elapsed=time.perf_counter()-t
    print(seed,p,'seconds',round(elapsed,3),'leaves/gap/PSD',check(x,y,.035,.35,.52,out),flush=True)
PY
```
- **Container Exit Code**: `0` (`16332 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public studies: False | public validator: False
31 19 seconds 4.448 leaves/gap/PSD (1202, 1.0189826404882751e-07, 1.7461407470365332e-09)
34 18 seconds 2.423 leaves/gap/PSD (1480, 4.014479355607392e-06, 1.600584409587609e-09)
35 19 seconds 3.054 leaves/gap/PSD (783, 4.472797332510936e-08, 1.837529223884883e-09)
36 19 FAIL Certificate requires 2264 leaves 3.949 s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=82714184b77073fd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_82714184 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`48189 ms` LLM)
- **Rationale**: Retest the failing seed and its paired neighbors. The challenger must reduce the leaf bottleneck without sacrificing certificate validity or the 18-second budget for the two slowest solves.
- **Alternatives Weighed**: `Increase coordinate-descent sweeps at every node` (rejected: It could tighten some leaf bounds, but does not improve split selection and risks consuming the two-study runtime budget.)
- **ATMS NOGOOD DECLARED**: Refuted `A split rule based only on fractional coefficients provides enough partition compaction for the tested connected paired-column designs.` | Witness: `Paired-column seed 36 required 2,264 leaves against the 2,048-leaf cap, while nearby seeds used as many as 2,043.` | Blanket Diagnosis: `Near-substitute columns can have similar fractional coefficients but very different conditional subcube bounds; fractionality alone does not distinguish those splits.` | Sublated Pivot: `Probe both children of candidate splits using the same convex envelope, while retaining exact leaf-bound checks for pruning.`

```bash
cp /holdout_task/coupled_interferometry_certified_l0.py /tmp/eq02_v02.py
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/coupled_interferometry_certified_l0.py')
s=p.read_text()
ix=s.index('    def _leaf(')
s=s[:ix]+'''    def _branch(self, state, center, frac):
        # Probe both children in parallel with proximal steps on the same
        # convex envelope. Probe bounds choose a split, not a pruning rule.
        free = np.flatnonzero(state == -1)
        score = np.minimum(frac[free], 1 - frac[free])
        ambiguous = free[score > 1e-5]
        if len(ambiguous) < 2:
            return int(ambiguous[0] if len(ambiguous) else free[np.argmax(np.abs(center[free]))]), None
        jj = ambiguous[np.argsort(-np.minimum(frac[ambiguous], 1 - frac[ambiguous]))[:8]]
        states = np.tile(state, (2 * len(jj), 1))
        states[np.arange(len(states)), np.repeat(jj, 2)] = np.tile([0, 1], len(jj))
        off = states == 0
        active = states == 1
        step = self.step
        slope, a, knee = self.slope, self.a, self.knee
        c = np.tile(center, (len(states), 1))
        c[off] = 0
        v = c.copy()
        for k in range(1, 26):
            z = v + step * (self.b - v @ self.q)
            mag = np.abs(z)
            penalized = np.maximum(0, mag - step * slope)
            quadratic = mag / (1 + step * a)
            prox = np.where(active | (penalized > knee), quadratic, penalized)
            nxt = np.copysign(np.minimum(self.box, prox), z)
            nxt[off] = 0
            v = nxt + ((k - 1) / (k + 2)) * (nxt - c)
            c = nxt
        lbs = np.array([self._bound(w, st) for w, st in zip(c, states)]).reshape(-1, 2)
        # A child already above the incumbent has no additional value.
        lbs = np.minimum(lbs, self.best - self.tol)
        rank = .8 * lbs.min(axis=1) + .2 * lbs.max(axis=1)
        i = int(np.argmax(rank))
        return int(jj[i]), c[2*i:2*i+2]

'''+s[ix:]
s=s.replace('        self.a = self.d + lam2','        self.a = self.d + lam2\n        self.step = 1 / max(np.linalg.eigvalsh(self.q)[-1], 1e-12)')
a=s.index('            score = np.where(free,')
b=s.index('            for value in ',a)
s=s[:a]+'            j, starts = self._branch(state, center, frac)\n'+s[b:]
s=s.replace('self._node(child, center)','self._node(child, center if starts is None else starts[value])')
p.write_text(s)
PY
cat > /tmp/paired_eq02.py <<'PY'
import sys,time
sys.path.insert(0,'/tmp')
from test_eq02 import eq,check,design
import numpy as np
for seed,p in [(36,19),(29,19),(33,18),(31,19),(34,18),(35,19)]:
    rng=np.random.default_rng(seed)
    a=design(rng,120,(p+1)//2,.10)
    x=np.repeat(a,2,axis=1)[:,:p]+.005*rng.normal(size=(120,p))
    y=a@(rng.choice([-1.,1.],a.shape[1])*rng.uniform(.47,.59,a.shape[1]))+.04*rng.normal(size=120)
    t=time.perf_counter()
    try:
        out=eq._Selector(x,y,.035,.35,.52).solve()
        dt=time.perf_counter()-t
        print(seed,p,round(dt,3),check(x,y,.035,.35,.52,out),flush=True)
    except RuntimeError as e:
        print(seed,p,round(time.perf_counter()-t,3),'FAIL',e,flush=True)
PY
python3 -I /tmp/paired_eq02.py
```
- **Container Exit Code**: `0` (`27848 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
36 19 5.955 (1495, 9.058123140448515e-08, 1.5116986748875699e-09)
29 19 4.128 (1140, 4.203052128559648e-09, 1.630945902658355e-09)
33 18 3.841 (1133, 2.9938720422784115e-06, 1.5540405724113469e-09)
31 19 4.536 (983, 2.8866386454495796e-06, 1.7461407470365332e-09)
34 18 2.24 (619, 1.6394778423745038e-08, 1.600584409587609e-09)
35 19 3.746 (720, 1.2677565131014035e-07, 1.837529223884883e-09)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e11a179ccebff6eb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_e11a179c (directory).
```

### Turn 10 — `execute_in_remote_docker` (`22525 ms` LLM)
- **Rationale**: Lookahead branching reduced the failing case from 2,264 to 1,495 leaves. Verify the current artifact in isolated Python on the end-to-end pipeline, exhaustive face oracle, and non-paired stress cases not yet run on this version. A regression would prevent submission; the paired-column results need no repeat.

```bash
python3 -I /tmp/test_eq02.py && python3 -I - <<'PY'
import sys, time
sys.path.insert(0, '/tmp')
from test_eq02 import eq, check, design
import numpy as np
for seed, p, rho in [(17, 19, .30), (18, 18, .30), (51, 19, .997)]:
    rng = np.random.default_rng(seed)
    x = design(rng, 120, p, rho)
    if rho < .9:
        g = x.T @ x
        z = rng.choice([-1., 1.], p) * np.sqrt(.7 * g.diagonal()) * rng.uniform(.88, 1.12, p)
        y = x @ np.linalg.solve(g, z) + .07 * rng.normal(size=120)
    else:
        w = np.zeros(p)
        w[:8] = rng.choice([-1., 1.], 8) * rng.uniform(1.1, 1.4, 8)
        y = x @ w + .055 * rng.normal(size=120)
    start = time.perf_counter()
    out = eq._Selector(x, y, .035, .35, .52).solve()
    elapsed = time.perf_counter() - start
    print(seed, p, 'seconds', round(elapsed, 4), 'leaves/gap/PSD', check(x, y, .035, .35, .52, out), flush=True)
    assert elapsed <= 9
PY
```
- **Container Exit Code**: `0` (`6143 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
face oracle error 0.0 leaves/gap/psd (3, 3.540662518730642e-08, 4.294711511082793e-05)
panel 91 18 0.34 MAE 0.040617 relL2 0.016898 leaves/gap/psd (2, 6.563260068759291e-08, 2.686373436008174e-06) seconds 1.2099
panel 92 19 0.72 MAE 0.047785 relL2 0.014953 leaves/gap/psd (3, 4.1811414774883815e-08, 9.92604307488153e-07) seconds 1.5839
panel 93 19 0.86 MAE 0.045117 relL2 0.013148 leaves/gap/psd (5, 3.932719749855096e-08, 5.37788130857206e-07) seconds 0.3812
panel 96 18 0.92 MAE 0.04613 relL2 0.015413 leaves/gap/psd (5, 4.456537538999328e-08, 2.036738236890544e-07) seconds 0.5785
17 19 seconds 0.1467 leaves/gap/PSD (8, 2.300915902253564e-08, 1.5110531429215819e-06)
18 18 seconds 0.2668 leaves/gap/PSD (26, 7.206166152684546e-08, 1.5581613862151613e-06)
51 19 seconds 0.2492 leaves/gap/PSD (36, 1.8632329457091146e-08, 6.742006863692847e-09)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e11a179ccebff6eb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_e11a179c (directory).
```

### Turn 11 — `submit_completed_task` (`14550 ms` LLM)
- **Rationale**: The verified candidate reduced the failing paired-column case from 2,264 to 1,495 leaves. Independent checks confirmed partition coverage and recomputed the leaf bounds; the exhaustive oracle matched the objective exactly. The isolated end-to-end and stress tests passed without further artifact changes.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:13507'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Isolated Python tests passed on four synthetic end-to-end studies, nine connected-design stress cases, and an exhaustive small-problem oracle. The largest certificate used 1,495 of 2,048 leaves, the worst optimality gap was 2.99e-6, and the two slowest solves totaled 10.49 s. The public studies and validator were unavailable, so official validation remains unverified.`

### Turn 12 — `execute_in_remote_docker` (`34073 ms` LLM)
- **Rationale**: Fit on other centers and compare held-out calibration residuals against OLS to detect a center-specific calibration failure that pooled performance could hide. Use the public measurements and validator if mounted; otherwise label the new audit synthetic and leave official validation unverified.
- **Alternatives Weighed**: `Audit relay modes, drift operators, or balanced classification accuracy` (rejected: This study has no relay, drift equation, or class outcome. Held-out-center assay residuals test the relevant nuisance mechanism.)

```bash
cat > /tmp/audit_eq02.py <<'PY'
import pickle
import runpy
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/tmp')
from test_eq02 import eq, check, panel_trial

root = Path('/holdout_task')
studies_path = root / 'public_eq02_studies.pkl'
validator = root / 'validate_public_schema.py'
print('Public studies present:', studies_path.is_file(), '| official validator present:', validator.is_file(), flush=True)
if validator.is_file():
    sys.path.insert(0, str(root))
    runpy.run_path(str(validator), run_name='__main__')

solve = eq.solve_coupled_interferometry_l0

def audit(study):
    panel = study['panel']
    centers = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    cal = np.asarray(study['calibration_indices'])
    optics = study['interferometry']
    phase = np.angle(optics['signal_phasor'] * np.conj(optics['ref_phasor']))
    h = np.column_stack([np.ones(len(strata)), strata == 1, strata == 2, eq._align(panel), phase])
    y = np.asarray(study['y_assay_raw'])
    if cal.dtype == bool:
        cal = np.flatnonzero(cal)
    folds = []
    for c in np.unique(centers[cal]):
        train, held = cal[centers[cal] != c], cal[centers[cal] == c]
        robust = eq._calibrate(h[train], y[train])
        ols = np.linalg.lstsq(h[train], y[train], rcond=None)[0]
        folds.append(tuple(float(np.median(np.abs(y[held] - h[held] @ beta))) for beta in (robust, ols)))
    print('Held-out-center calibration median absolute residual (Cauchy, OLS):', np.round(folds, 5).tolist(), flush=True)
    start = time.perf_counter()
    out = solve(study)
    elapsed = time.perf_counter() - start
    result = check(study['X_cand'], out['deconfounded_target'], study['lam2'], study['lam0'], study['M'], out['certified_selection'])
    print('Clean-process solve seconds:', round(elapsed, 4), '| leaves/gap/minimum PSD eigenvalue:', result, flush=True)
    return out

if studies_path.is_file():
    with studies_path.open('rb') as fh:
        studies = pickle.load(fh)
    if isinstance(studies, dict):
        studies = [studies] if 'panel' in studies else studies.values()
    for study in studies:
        audit(study)
else:
    print('Running the requested center audit on synthetic data; no public measurements are mounted.', flush=True)
    eq.solve_coupled_interferometry_l0 = audit
    panel_trial(107, 19, .83)
PY
python3 -I /tmp/audit_eq02.py
```
- **Container Exit Code**: `0` (`2969 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public studies present: False | official validator present: False
Running the requested center audit on synthetic data; no public measurements are mounted.
Held-out-center calibration median absolute residual (Cauchy, OLS): [[0.10662, 0.36647], [0.15368, 0.39237], [0.10656, 0.37074], [0.09718, 0.40157], [0.14062, 0.31623]]
Clean-process solve seconds: 1.3965 | leaves/gap/minimum PSD eigenvalue: (6, 4.92946151453566e-07, 4.062348971233221e-06)
panel 107 19 0.83 MAE 0.042343 relL2 0.014296 leaves/gap/psd (6, 4.92946151453566e-07, 4.062348971233221e-06) seconds 2.0254

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e11a179ccebff6eb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_e11a179c (directory).
```

### Turn 13 — `execute_in_remote_docker` (`21452 ms` LLM)
- **Rationale**: The latest center audit did not measure the leaf-count failure tracked by ng_t09. Print the failing seed’s verified leaf count and gap explicitly against their limits in isolated Python. A failed check blocks submission; a pass closes the outstanding probe without changing the artifact.

```bash
python3 -I - <<'PY'
import importlib.util
import sys
import time
import numpy as np
sys.path.insert(0, '/tmp')
from test_eq02 import eq, check, design

rng = np.random.default_rng(36)
a = design(rng, 120, 10, .10)
x = np.repeat(a, 2, axis=1)[:, :19] + .005 * rng.normal(size=(120, 19))
y = a @ (rng.choice([-1., 1.], 10) * rng.uniform(.47, .59, 10)) + .04 * rng.normal(size=120)
start = time.perf_counter()
out = eq._Selector(x, y, .035, .35, .52).solve()
elapsed = time.perf_counter() - start
leaves, gap, psd = check(x, y, .035, .35, .52, out)
print(f'ng_t09 paired-column seed=36 p=19 baseline_leaves=2264 candidate_leaves={leaves} leaf_limit=2048 leaf_headroom={2048-leaves} certified_gap={gap:.12g} gap_limit=1e-5 minimum_psd_eigenvalue={psd:.12g} solve_seconds={elapsed:.4f}')
assert leaves <= 2048 and 0 <= gap <= 1e-5
PY
```
- **Container Exit Code**: `0` (`6341 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ng_t09 paired-column seed=36 p=19 baseline_leaves=2264 candidate_leaves=1495 leaf_limit=2048 leaf_headroom=553 certified_gap=9.05812314045e-08 gap_limit=1e-5 minimum_psd_eigenvalue=1.51169867489e-09 solve_seconds=4.7898

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e11a179ccebff6eb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_e11a179c (directory).
```

### Turn 14 — `submit_completed_task` (`16758 ms` LLM)
- **Rationale**: The saved candidate passed isolated verification, including full partition coverage and independent leaf-bound recomputation on the failing slice. Cauchy calibration also beat OLS on median held-out residuals in all five synthetic centers. Further reruns would not resolve the missing official fixtures.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Isolated Python tests passed on five synthetic end-to-end studies, nine connected-design stress cases, and an exhaustive small-problem oracle. The largest certificate used 1,495 of 2,048 leaves, the worst certified gap was 2.99e-6, and the two slowest solves totaled 10.49 seconds. The public studies and official validator were not mounted, so official validation remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_interferometry_certified_l0.py": "FILE:13507"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_panel_mae": 0.045663,
    "max_calibration_rel_l2": 0.018845,
    "max_deconfounded_mae": 0.201469,
    "max_stage3_w_vs_clean_rel_l2": 0.171281,
    "max_duality_gap": 2.21e-07,
    "student_elapsed_sec": 0.0874,
    "instance_diagnostics": [
      {
        "seed": 99411,
        "p": 18,
        "panel_mae": 0.045663,
        "cal_rel_l2": 0.016941,
        "deconf_mae": 0.177163,
        "w_vs_clean_rel_l2": 0.171281,
        "cert_ok": true,
        "cert_reason": "ok"
      },
      {
        "seed": 99412,
        "p": 19,
        "panel_mae": 0.045333,
        "cal_rel_l2": 0.018845,
        "deconf_mae": 0.201469,
        "w_vs_clean_rel_l2": 0.094931,
        "cert_ok": true,
        "cert_reason": "ok"
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.018845, \"max_deconfounded_mae\": 0.201469, \"max_stage3_w_vs_clean_rel_l2\": 0.171281, \"max_duality_gap\": 2.21e-07, \"student_elapsed_sec\": 0.0874, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.016941, \"deconf_mae\": 0.177163, \"w_vs_clean_rel_l2\": 0.171281, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.018845, \"deconf_mae\": 0.201469, \"w_vs_clean_rel_l2\": 0.094931, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n__HEGEL_VERIFIER_AUTH_eeb7de76405825bbb628b58b7f768c53__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.018845, \"max_deconfounded_mae\": 0.201469, \"max_stage3_w_vs_clean_rel_l2\": 0.171281, \"max_duality_gap\": 2.21e-07, \"student_elapsed_sec\": 0.0874, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.016941, \"deconf_mae\": 0.177163, \"w_vs_clean_rel_l2\": 0.171281, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.018845, \"deconf_mae\": 0.201469, \"w_vs_clean_rel_l2\": 0.094931, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5088.86,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
