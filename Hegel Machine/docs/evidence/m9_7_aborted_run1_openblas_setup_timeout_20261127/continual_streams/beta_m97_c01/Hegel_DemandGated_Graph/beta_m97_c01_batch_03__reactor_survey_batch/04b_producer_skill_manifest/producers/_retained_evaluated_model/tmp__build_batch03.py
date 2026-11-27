from pathlib import Path
p = Path('/tmp/hegel_continual_archive/beta_m97_c01_batch_02_solver.py')
s = p.read_text().replace('Open-Robin reactor tomography', 'Periodic recirculation reactor tomography').replace('from scipy.sparse import diags, eye, kron', 'from scipy.sparse import coo_matrix, diags, eye, kron')
a, b = s.index('def _calibrated_target'), s.index('def _axis_operator')
s = s[:a] + '''def _bank_sweep(sweeps, bank):
    rows = np.asarray(sweeps['bank_id']) == bank
    return (np.asarray(sweeps['ref_signal'], dtype=float)[rows],
            np.asarray(sweeps['raw_readout'], dtype=float)[rows])


def _clean_sweep(x, y):
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    distinct = dx != 0.0
    if not np.any(distinct):
        return np.ones(x.size, dtype=bool)
    slope = np.median((y[j] - y[i])[distinct] / dx[distinct])
    offsets = y - slope * x
    return np.abs(offsets - np.median(offsets)) <= 3.0


def _calibrated_target(study):
    sweeps = study['sensor_calibration_sweeps']
    row_banks = np.asarray(study['row_bank_id'])
    obs = np.asarray(study['y_obs_raw'], dtype=float)
    result = np.empty_like(obs)
    archive = None
    for bank in range(4):
        x, y = _bank_sweep(sweeps, bank)
        clean = _clean_sweep(x, y)
        count = np.count_nonzero(clean)
        if count >= 4 or (count >= 2 and np.all(clean)):
            gain, offset = _fit_line(x[clean], y[clean])
        else:
            if archive is None:
                archive = json.loads(_CALIBRATION_ARCHIVE.read_text())
            for record in reversed(archive):
                xp, yp = _bank_sweep(record, bank)
                keep = _clean_sweep(xp, yp)
                if np.count_nonzero(keep) >= 2:
                    gain, offset = _fit_line(xp[keep], yp[keep])
                    break
            else:
                raise ValueError(f'Insufficient calibration data for bank {bank}')
        rows = row_banks == bank
        result[rows] = (obs[rows] - offset) / gain
    return result


def _periodic_axis(kappa, velocity, length):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    h = length / (n - 1)
    i = np.arange(n)
    im, ip = (i - 1) % n, (i + 1) % n
    km = (kappa + kappa[im]) / (2 * h**2)
    kp = (kappa + kappa[ip]) / (2 * h**2)
    data = np.r_[-km - np.maximum(velocity, 0) / h,
                 -kp - np.maximum(-velocity, 0) / h,
                 km + kp + np.abs(velocity) / h]
    return coo_matrix((data, (np.tile(i, 3), np.r_[im, ip, i])), shape=(n, n)).tocsc()


''' + s[b:]
a, b = s.index('    ax =', s.index('def _forward')), s.index('    operator =', s.index('def _forward'))
s = s[:a] + '''    mode = float(study['recirculation_loop_mode'])
    potential = np.asarray(study['V_reaction_2d'], dtype=float)
    if mode > 0.50:
        ax = _periodic_axis(study['kappa_x'], study['vel_x'], study['Lx'])
        ay = _periodic_axis(study['kappa_y'], study['vel_y'], study['Ly'])
        cx = np.cos(2 * np.pi * np.linspace(0.0, 1.0, nx))
        cy = np.cos(2 * np.pi * np.linspace(0.0, 1.0, ny))
        potential = potential + 0.75 * mode * cx[:, None] * cy[None, :]
    else:
        ax = _axis_operator(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
        ay = _axis_operator(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    potential = potential.ravel(order='F')
''' + s[b:]
a, b = s.index('def solve_'), s.index('    target = _calibrated_target')
s = s[:a] + 'def solve_beta_m97_c01_batch_03(study: dict) -> dict:\n' + s[b:]
Path('/holdout_task/beta_m97_c01_batch_03_solver.py').write_text(s)
