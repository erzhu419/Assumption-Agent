from pathlib import Path
archive = Path('/tmp/hegel_continual_archive/beta_m96_w01_batch_03_solver.py').read_text()
cal = archive[archive.index('def _calibrate('):archive.index('def _forward(')]
cal = cal.replace("            raise ValueError(f'Bank {b} needs calibration at two distinct reference levels')", "            gain, offset = _prior_bank(b)\n            rows = row_bank == b\n            calibrated[rows] = (y[rows] - offset) / gain\n            continue")
head = '''"""Calibrated multi-wing tracer tomography with circulation memory."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _prior_bank(bank):
    with _HISTORY.open() as handle:
        history = json.load(handle)
    for sweep in reversed(history):
        mask = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'])[mask]
        y = np.asarray(sweep['raw_readout'])[mask]
        if x.size >= 2 and np.ptp(x) > 0:
            return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
    raise ValueError(f'No identifiable calibration for bank {bank}')


'''
forward = '''def _forward(study):
    src = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = src.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    interaction = np.linalg.norm(v - vx[:, None] - vy[None, :])
    norm = np.linalg.norm(v)
    eddy = (interaction > 1e-4 * norm or
            np.any(np.asarray(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']))
    factor = 1.45 if eddy else 1.0
    ax = _build_1d(study['Lx'], nx, factor * np.asarray(study['kappa_x']),
                   study['vel_x'], study['robin_x'], study['reaction_x'])
    by = _build_1d(study['Ly'], ny, factor * np.asarray(study['kappa_y']),
                   study['vel_y'], study['robin_y'], study['reaction_y'])
    shifts = study['mu_shifts']
    si, sj = np.asarray(study['sensor_ij']).T
    blocks = []
    if eddy or interaction > 1e-6 * norm:
        # Keep the exact reaction in the threshold interval without forcing eddy diffusion.
        base = (kron(csc_matrix(ax), eye(ny), format='csc') +
                kron(eye(nx), csc_matrix(by), format='csc') + diags(v.ravel(), format='csc'))
        ident = eye(nx * ny, format='csc')
        rhs = src.reshape(m, -1).T
        for mu in shifts:
            fields = splu(base + mu * ident).solve(rhs).T.reshape(src.shape)
            blocks.append(fields[:, si, sj].T)
    else:
        a, qx = schur(ax + np.diag(vx), output='real')
        b, qy = schur(by + np.diag(vy), output='real')
        rhs = (qx.T @ src) @ qy
        for mu in shifts:
            shifted = a + mu * np.eye(nx)
            fields = np.empty_like(src)
            for j, load in enumerate(rhs):
                u, scale, info = dtrsyl(shifted, b, load, tranb='T')
                if info:
                    raise np.linalg.LinAlgError('Ill-conditioned shifted transport operator')
                fields[j] = (qx @ (u / scale)) @ qy.T
            blocks.append(fields[:, si, sj].T)
    return np.concatenate(blocks, axis=0), fields


'''
tail = archive[archive.index('def _residualize('):].replace('solve_beta_m96_w01_s00','solve_beta_m96_w01_batch_04')
Path('/holdout_task/beta_m96_w01_batch_04_solver.py').write_text(head + cal + forward + tail)
