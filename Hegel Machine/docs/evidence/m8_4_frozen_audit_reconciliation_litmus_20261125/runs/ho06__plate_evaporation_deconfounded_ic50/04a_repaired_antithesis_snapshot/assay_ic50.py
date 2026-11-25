import re

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.special import expit


_LN10 = np.log(10.0)
_WELL_LABEL = re.compile(r'([A-Pa-p])[-_ ]*([0-9]{1,2})')


def _position(well):
    row = next((well[k] for k in ('row', 'row_index', 'r') if k in well), None)
    col = next((well[k] for k in ('col', 'column', 'col_index', 'c') if k in well), None)
    if row is not None and col is not None:
        r = str(row).strip().upper()
        if len(r) == 1 and 'A' <= r <= 'P':
            return ord(r) - ord('A') + 1, int(col), True, False
        return int(row), int(col), False, False
    label = next((well[k] for k in ('well', 'well_id', 'position') if k in well), '')
    if isinstance(label, (tuple, list)):
        return _position({'row': label[0], 'col': label[1]})
    match = _WELL_LABEL.fullmatch(str(label).strip())
    if match is None:
        raise ValueError('A well needs a plate row and column or an A01-style position.')
    return ord(match[1].upper()) - ord('A') + 1, int(match[2]), True, True


def _spatial_basis():
    # Center spatial terms over the physical grid, not over the compound layout.
    r, c = np.indices((16, 24), dtype=float)
    x, y = 2 * r / 15 - 1, 2 * c / 23 - 1
    dr, dc = np.minimum(r, 15 - r), np.minimum(c, 23 - c)
    edge = np.minimum(dr, dc)
    px = (np.ones_like(x), x, (3*x*x - 1)/2, (5*x**3 - 3*x)/2,
          (35*x**4 - 30*x*x + 3)/8)
    py = (np.ones_like(y), y, (3*y*y - 1)/2, (5*y**3 - 3*y)/2,
          (35*y**4 - 30*y*y + 3)/8)
    fields = []

    def add(field, amplitude):
        fields.append(amplitude * (field - field.mean()))

    # Broad illumination terms, separable and nonseparable.
    for degree, amplitude in ((1, .22), (2, .11), (3, .055), (4, .035)):
        for i in range(degree + 1):
            add(px[i] * py[degree - i], amplitude)

    # Perimeter and axis-specific falloff, from a sharp rim to a broad band.
    for length in (.45, 1.35, 3.5):
        add(np.exp(-edge / length), .17)
    for length in (.85, 2.5):
        er, ec = np.exp(-dr / length), np.exp(-dc / length)
        add(er, .09)
        add(ec, .09)
        add(er * ec, .09)
        add(x * er, .07)
        add(y * ec, .07)
    band = np.exp(-edge / 1.65)
    add(x * band, .065)
    add(y * band, .065)
    add(x * y * band, .05)
    return np.stack(fields, axis=-1)


def _plate_records(plate):
    positions = [_position(w) for w in plate['wells']]
    if not positions:
        return []
    # Full-grid numeric positions can be indexed from zero or one. Letter
    # rows and standard well labels explicitly carry one-based coordinates.
    row_base = int(any(p[2] or p[0] == 16 for p in positions)
                   and not any(p[0] == 0 for p in positions))
    col_base = int(any(p[3] or p[1] == 24 for p in positions)
                   and not any(p[1] == 0 for p in positions))
    records = []
    for w, (r, c, letter_row, label) in zip(plate['wells'], positions):
        kind = w['well_type']
        if kind not in ('DMSO', 'POS_KILL', 'COMPOUND'):
            continue
        r -= 1 if letter_row else row_base
        c -= 1 if label else col_base
        if not (0 <= r < 16 and 0 <= c < 24):
            raise ValueError('Well position is outside the 16 by 24 plate.')
        dose = float(w['log10_conc_uM']) if kind == 'COMPOUND' else 0.0
        cid = w['compound_id'] if kind == 'COMPOUND' else None
        records.append((r, c, kind, dose, float(w['fluorescence']), cid))
    records.sort(key=lambda w: w[:5])
    return records


def _initial_gain(design, values):
    # The spatial columns already encode the shrinkage scales. Fit the DMSO
    # surface with pseudo-Huber IRLS, leaving the mean gain unpenalized.
    n_terms = design.shape[1]
    prior = .025 * np.eye(n_terms)[1:]
    beta = np.zeros(n_terms)
    beta[0] = np.median(values)
    target = np.r_[values, np.zeros(n_terms - 1)]
    for _ in range(12):
        resid = design @ beta - values
        root_weight = (1 + (resid / .04)**2)**(-.25)
        lhs = np.vstack((design * root_weight[:, None], prior))
        rhs = target.copy()
        rhs[:len(values)] *= root_weight
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return beta


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Fit shared compound Hill curves after plate-specific spatial calibration."""
    records = [_plate_records(plate) for plate in plates]
    records = [plate for plate in records if plate]
    records.sort(key=lambda plate: tuple(w[:5] for w in plate))
    basis = _spatial_basis()
    plate_size = basis.shape[-1] + 2  # background, mean gain, spatial gain
    n_plate_params = len(records) * plate_size
    ids = {}
    for plate in records:
        for w in plate:
            if w[2] == 'COMPOUND' and w[5] not in ids:
                ids[w[5]] = len(ids)
    if not ids:
        return {}

    n_params = n_plate_params + 2 * len(ids)
    start = np.zeros(n_params)
    lower = np.full(n_params, -np.inf)
    upper = np.full(n_params, np.inf)
    blocks = []
    scaled_readout, doses, indices, provisional, dmso_flags = [], [], [], [], []
    offset = 0
    for i, plate in enumerate(records):
        r, c, kind, x, f, cid = zip(*plate)
        dmso = np.asarray(kind) == 'DMSO'
        kill = np.asarray(kind) == 'POS_KILL'
        f = np.asarray(f)
        baseline = np.median(f[kill])
        signal = np.median(f[dmso]) - baseline
        if not np.isfinite(signal) or signal <= 0:
            raise ValueError('Plate controls must have positive DMSO-to-POS_KILL contrast.')
        y = (f - baseline) / signal
        mat = np.column_stack((np.ones(len(plate)), basis[r, c]))
        cols = slice(i * plate_size, (i + 1) * plate_size)
        beta = _initial_gain(mat[dmso], y[dmso])
        beta[0] = max(beta[0], .06)
        start[cols][1:] = beta
        lower[cols.start + 1] = .05
        rows = slice(offset, offset + len(plate))
        blocks.append((rows, cols, mat))
        offset = rows.stop
        scaled_readout.extend(y)
        doses.extend(x)
        indices.extend(ids[w[5]] if w[2] == 'COMPOUND' else -1 for w in plate)
        provisional.extend(y / np.maximum(mat @ beta, .25))
        dmso_flags.extend(dmso)

    y = np.asarray(scaled_readout)
    dose = np.asarray(doses)
    group = np.asarray(indices)
    provisional = np.asarray(provisional)
    controls = (group < 0)
    compounds = np.flatnonzero(~controls)
    dmso_flags = np.asarray(dmso_flags, dtype=float)
    n = len(y)
    for j in range(len(ids)):
        mask = group == j
        xj, vj = dose[mask], provisional[mask]
        levels = np.unique(xj)
        medians = np.array([np.median(vj[xj == x]) for x in levels])
        ic = levels[np.argmin(np.abs(medians - .5))]
        margin = max(4., .5 * (levels[-1] - levels[0]))
        lo, hi = [levels[0] - margin, .02], [levels[-1] + margin, 20.]
        init = least_squares(
            lambda p: expit(_LN10 * p[1] * (p[0] - xj)) - vj,
            [ic, 1.], bounds=(lo, hi), loss='soft_l1', f_scale=.05,
            max_nfev=100,
        )
        col = n_plate_params + 2 * j
        start[col:col+2] = init.x
        lower[col:col+2], upper[col:col+2] = lo, hi

    # Sparse joint fit: each observation touches one plate and at most one
    # compound. Controls fix the zero and unit viability anchors exactly.
    cp = n_plate_params + 2 * group[compounds]
    spatial_cols = np.array([k for k in range(n_plate_params) if k % plate_size >= 2])
    m = len(spatial_cols)
    jr, jc = [], []
    for rows, cols, _ in blocks:
        jr.append(np.repeat(np.arange(rows.start, rows.stop), plate_size))
        jc.append(np.tile(np.arange(cols.start, cols.stop), rows.stop - rows.start))
    jr.extend((compounds, compounds, n + np.arange(m)))
    jc.extend((cp, cp + 1, spatial_cols))
    jr, jc = np.concatenate(jr), np.concatenate(jc)
    dc = dose[compounds]
    weight = np.where(controls, 1.5, 1.)
    sigma = np.full(n, .025)

    def predict(p):
        v = dmso_flags.copy()
        delta = p[cp] - dc
        v[compounds] = expit(_LN10 * p[cp + 1] * delta)
        gain = np.empty(n)
        pred = np.empty(n)
        for rows, cols, mat in blocks:
            gain[rows] = mat @ p[cols][1:]
            pred[rows] = p[cols.start] + gain[rows] * v[rows]
        return pred, v, gain, delta

    def residual(p):
        return np.r_[(predict(p)[0] - y) / sigma, p[spatial_cols]]

    def jacobian(p):
        _, v, gain, delta = predict(p)
        pieces = []
        for rows, _, mat in blocks:
            local = np.column_stack((np.ones(len(mat)), v[rows, None] * mat))
            pieces.append((local / sigma[rows, None]).ravel())
        vc = v[compounds]
        d = (_LN10 * vc * (1 - vc) * gain[compounds]) / sigma[compounds]
        pieces.extend((d * p[cp + 1], d * delta, np.ones(m)))
        return csr_matrix((np.concatenate(pieces), (jr, jc)), shape=(n + m, n_params))

    def robust_loss(width, cauchy):
        width2 = width * width

        def loss(z):
            # Downweight readout failures, not the quadratic spatial prior.
            out = np.empty((3, len(z)))
            t = 1 + z[:n] / width2
            if cauchy:
                out[0, :n] = width2 * np.log1p(z[:n] / width2)
                out[1, :n] = 1 / t
                out[2, :n] = -1 / (width2 * t * t)
            else:
                root = np.sqrt(t)
                out[0, :n] = 2 * z[:n] / (root + 1)
                out[1, :n] = 1 / root
                out[2, :n] = -.5 / (width2 * t * root)
            out[:, :n] *= weight
            out[0, n:], out[1, n:], out[2, n:] = z[n:], 1., 0.
            return out

        return loss

    def fit(initial, loss):
        return least_squares(
            residual, initial, jac=jacobian, bounds=(lower, upper), loss=loss,
            tr_solver='lsmr', tr_options={'atol': 1e-8, 'btol': 1e-8},
            ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=160,
        ).x

    first = fit(start, robust_loss(1.5, False))
    raw = predict(first)[0] - y
    for rows, _, _ in blocks:
        e = raw[rows]
        sigma[rows] = max(.005, 1.4826 * np.median(np.abs(e - np.median(e))))
    fitted = fit(first, robust_loss(2.8, True))
    return {
        cid: {
            'log10_ic50_uM': float(fitted[n_plate_params + 2*j]),
            'hill_slope': float(fitted[n_plate_params + 2*j + 1]),
        }
        for cid, j in ids.items()
    }
