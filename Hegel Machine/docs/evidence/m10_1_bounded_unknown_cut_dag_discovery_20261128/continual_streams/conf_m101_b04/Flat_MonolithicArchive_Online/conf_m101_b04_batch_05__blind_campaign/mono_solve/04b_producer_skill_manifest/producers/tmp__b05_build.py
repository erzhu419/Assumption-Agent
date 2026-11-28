import ast
from pathlib import Path
src=Path('/tmp/hegel_continual_archive/conf_m101_b04_batch_04_solver.py').read_text()
mod=ast.parse(src)
funcs={n.name:n for n in mod.body if isinstance(n,ast.FunctionDef)}
needed={'_calibrate'}
while True:
    new=needed|{n.id for k in needed for n in ast.walk(funcs[k]) if isinstance(n,ast.Name) and n.id in funcs}
    if new==needed: break
    needed=new
cal='\n\n'.join(ast.get_source_segment(src,n) for n in mod.body if isinstance(n,ast.FunctionDef) and n.name in needed)
body='''

def _panel_b(panel, x):
    u, v = panel['u'], panel['v']

    def features(q, p):
        return _affine(q + p[1] * np.sin(p[0] * q))

    def residual(p):
        a = features(u, p)
        return (a @ _ls(a, v) - v).ravel()

    p = least_squares(residual, [2.85, -.4], bounds=([1.5, -1.], [4.5, -.05])).x
    return features(x, p) @ _ls(features(u, p), v)


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _scalar_features(z):
    return np.column_stack((_affine(z), z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z = _panel_b(study['panel_b'], _calibrate(study['panel_a'], x))
    a = _scalar_features(z)
    y = a[n:] @ _ls(a[:n], train['y'])
    a = _affine(z)
    state = a[n:] @ _ls(a[:n], train['traj'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
'''
imports='\n'.join(ast.get_source_segment(src,n) for n in mod.body if isinstance(n,(ast.Import,ast.ImportFrom)))
if 'least_squares' not in imports: imports+='\nfrom scipy.optimize import least_squares'
Path('/holdout_task/batch_solver.py').write_text('"""Invert the reference calibration and fit readouts of the panel B operator."""\n'+imports+'\n\n'+cal+body)
