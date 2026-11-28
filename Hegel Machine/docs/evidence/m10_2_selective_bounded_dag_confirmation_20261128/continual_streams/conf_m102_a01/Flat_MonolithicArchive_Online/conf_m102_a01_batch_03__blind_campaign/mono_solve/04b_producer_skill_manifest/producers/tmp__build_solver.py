from pathlib import Path
src=Path('/tmp/hegel_continual_archive/conf_m102_a01_batch_00_solver.py').read_text()
src=src.replace('"""Calibrated nonlinear prediction with a shared low-rank time response."""','"""Panel-informed covariance models and a shared low-rank trajectory predictor."""')
src=src[:src.index('\ndef solve_batch(')]
gp=Path('/tmp/gp.py').read_text()
blocks=gp[gp.index('def blocks('):gp.index('def predict(')]
blocks=blocks.replace('def blocks(tr,mode):','def _covariance_blocks(x, coordinates, n, mode):').replace(' K=[]',' tr = slice(0, n)\n Ws = coordinates\n K=[]',1).replace('q[np.ix_(tr,tr)]','q[tr,tr]').replace('ids in comb(','ids in combinations_with_replacement(')
src += '\n'+blocks+'''
def _gram_prediction(x, blocks, response, penalty):
    n = len(response)
    eye = np.eye(n)
    train_components = np.concatenate((blocks[:, :n, :n], eye[None]))
    flat = np.ascontiguousarray(train_components.reshape(len(train_components), -1))
    base = 2 + 2 * (x @ x[:n].T)

    def covariance(weights):
        return base[:n] + np.einsum('i,ij->j', weights, flat).reshape(n, n)

    def objective(theta):
        weights = np.exp(theta)
        factor = cho_factor(covariance(weights), check_finite=False)
        alpha = cho_solve(factor, response, check_finite=False)
        inv = cho_solve(factor, eye, check_finite=False)
        regularizer = penalty * weights[:-1]**.35
        value = .5 * (response @ alpha) + np.log(factor[0].diagonal()).sum() + regularizer.sum()
        grad = .5 * weights * np.einsum('ij,j->i', flat, (inv - np.outer(alpha, alpha)).ravel())
        grad[:-1] += .35 * regularizer
        return value, grad

    initial = np.full(len(flat), -7.)
    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   bounds=[(-16, 1)] * len(blocks) + [(-13, -2)],
                   options={'maxiter': 120, 'ftol': 1e-8})
    weights = np.exp(fit.x)
    factor = cho_factor(covariance(weights), check_finite=False)
    cross = base[n:] + np.einsum('i,ijk->jk', weights[:-1], blocks[:, n:, :n])
    return cross @ cho_solve(factor, response, check_finite=False)


def _panel_coordinates(study, x):
    z, h = _coordinates(study, x)
    panel = study['panel_b']
    b = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    coordinates = [x, z, h, _affine(z) @ b, _affine(x) @ b]
    panel = study['panel_c']
    for w in (x, z, h):
        for key in ('v1', 'v2'):
            def basis(a):
                transformed = a / (1 + .84 * abs(a)) if key == 'v1' else np.sqrt(1 + .959**2 * a*a)
                return np.column_stack((_affine(a), transformed))
            c = np.linalg.lstsq(basis(panel['u']), panel[key], rcond=None)[0]
            coordinates.append(basis(w) @ c)
    return coordinates


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    response = np.asarray(train['y'], dtype=float)
    n = len(response)
    poly, smooth, vt = _components(study)
    predictions = [.65 * poly[:, 0] + .35 * smooth[:, 0]]
    coords = _panel_coordinates(study, x)
    for mode, penalty in (('iso', 0.), ('full', 3.), ('all', 0.)):
        blocks = _covariance_blocks(x, coords, n, mode)
        predictions.append(_gram_prediction(x, blocks, response, penalty))
    y = np.mean(predictions, axis=0)
    state = (.4 * poly[:, 1:] + .6 * smooth[:, 1:]) @ vt
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
'''
Path('/holdout_task/batch_solver.py').write_text(src)
