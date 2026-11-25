from pathlib import Path

artifact = Path('/holdout_task/multicenter_calibration.py').read_text()
stress_path = '/tmp/assay_stress.py'
stress = Path(stress_path).read_text()
load = 'spec.loader.exec_module(assay)'
anchor = '            fitted_loc[i, m] = fitted\n'
assert artifact.count(anchor) == 1 and stress.count(load) == 1
for pseudo_count in (0, 6, 16):
    candidate = artifact
    if pseudo_count:
        candidate = candidate.replace(anchor, anchor + f'''            # A single control gives weak location evidence, but no spread estimate.
            for j in np.flatnonzero(n[i] == 1):
                col = raw[group == centers[i], j]
                value = col[np.isfinite(col)][0]
                fitted_loc[i, j] += (value - fitted_loc[i, j]) / (1 + {pseudo_count})
''')
    print('singleton location prior mass', pseudo_count, flush=True)
    exec(compile(stress.replace(load, 'exec(compile(candidate, "candidate", "exec"), assay.__dict__)'), stress_path, 'exec'), {'candidate': candidate})
