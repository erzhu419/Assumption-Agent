"""Calibrate coupled sensors and recover the downstream coordinates."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _linear_fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _shape(x, name, k):
    if name == 'sin':
        return np.sin(k*x)/k-x
    if name == 'tanh':
        return np.tanh(k*x)/k-x
    if name == 'atan':
        return np.arctan(k*x)/k-x
    if name == 'rat':
        return x/(1+k*x*x)-x
    return x**3


def _shape_derivative(x, name, k):
    if name == 'sin':
        return np.cos(k*x)-1
    if name == 'tanh':
        return -np.tanh(k*x)**2
    if name == 'atan':
        return 1/(1+(k*x)**2)-1
    if name == 'rat':
        return (1-k*x*x)/(1+k*x*x)**2-1
    return 3*x*x


def _fit_sensor(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.max(np.abs(u-v), axis=1) < 2
    u, v = u[keep], v[keep]
    models = []
    for j in range(4):
        a, b, y = u[:,j], u[:,(j+2)%4], v[:,j]
        best = None
        for f in ('sin','tanh','atan','rat','cube'):
            for g in ('sin','tanh','atan','rat','cube'):
                def design(k):
                    return np.column_stack((np.ones(len(a)), a, b, _shape(a,f,k[0]), _shape(b,g,k[1])))
                def residual(k):
                    d = design(k)
                    return d @ _linear_fit(d,y)-y
                k = least_squares(residual, [1.3, 2.], bounds=(.08, 4.5), max_nfev=75).x
                d = design(k)
                co = _linear_fit(d,y)
                r = d@co-y
                n = len(a)
                score = n*np.log(max(np.mean(r*r),1e-30))+(5+(f!='cube')+(g!='cube'))*np.log(n)
                if best is None or score < best[0]:
                    best = (score, f, g, k, co)
        models.append(best[1:])
    return models


def _undo_sensor(x, models):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for j in range(2):
        cols = [j,j+2]
        pair = [models[i] for i in cols]
        for p, obs in enumerate(x[:,cols]):
            def evaluate(q, jac=False):
                z = np.empty((2,2) if jac else (2,))
                for i, (f,g,k,co) in enumerate(pair):
                    a,b=q[i],q[1-i]
                    if jac:
                        z[i,i] = co[1]+co[3]*_shape_derivative(a,f,k[0])
                        z[i,1-i] = co[2]+co[4]*_shape_derivative(b,g,k[1])
                    else:
                        z[i] = co[0]+co[1]*a+co[2]*b+co[3]*_shape(a,f,k[0])+co[4]*_shape(b,g,k[1])-obs[i]
                return z
            out[p,cols] = least_squares(evaluate, obs, jac=lambda q:evaluate(q,True), max_nfev=45, gtol=1e-11).x
    return out


def _map_coordinates(panel, x, kind='sin', field='v'):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', field))
    def design(a, k):
        one = np.ones(len(a))
        if kind=='linear': return np.column_stack((one,a))
        if kind=='quad': return np.column_stack((one,a,a*a))
        if kind=='cube': return np.column_stack((one,a,a**3))
        return np.column_stack((one,a,_shape(a,kind,k)))
    k = 1.
    if kind not in ('linear','quad','cube'):
        def loss(k):
            d = design(u,k)
            return np.sum((d@_linear_fit(d,v)-v)**2)
        k = minimize_scalar(loss, bounds=(.12,4.5), method='bounded').x
    return design(x,k) @ _linear_fit(design(u,k),v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train']['x'], study['test']['x']
    z = _undo_sensor(np.concatenate((train,test)), _fit_sensor(study['panel_a']))
    c = _map_coordinates(study['panel_b'],z)
    return {'c': c[:len(train)], 'query': c[len(train):]}
