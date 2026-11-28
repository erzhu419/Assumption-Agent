import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel_features(panel, z):
    u = panel['u']
    features = []
    for key, bounds, nonlinear in (
        ('v1', (.1, 2.), lambda x,k: x/(1+k*abs(x))),
        ('v2', (.02, 1.), lambda x,k: x*x/(1+k*x*x)),
    ):
        def fit(k):
            a = np.column_stack((_linear(u), nonlinear(u,k)))
            coef = np.linalg.lstsq(a,panel[key],rcond=None)[0]
            return np.mean((a@coef-panel[key])**2), coef
        k = minimize_scalar(lambda k: fit(k)[0], bounds=bounds, method='bounded').x
        features.append(np.column_stack((_linear(z), nonlinear(z,k)))@fit(k)[1])
    return features


def _kernel_predict(q, target, rate, ridge):
    n = len(target)
    q = (q-q[:n].mean(0))/np.maximum(q[:n].std(0),1e-8)
    distance = rate*cdist(q,q[:n])/np.sqrt(q.shape[1])
    kernel = (1+distance+distance**2/3)*np.exp(-distance)
    offset = target.mean(0)
    return offset+kernel[n:]@np.linalg.solve(kernel[:n]+ridge*np.eye(n),target-offset)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z = inputs['node_02']['z']
    train = study['train']
    n = len(train['y'])
    x = np.vstack((train['x'],study['test']['x']))
    f = inputs['node_00']['f']
    b = f+(z-x)@np.linalg.lstsq(_linear(x),f,rcond=None)[0][1:]
    c,d = _panel_features(study['panel_c'],z)
    yh = _kernel_predict(np.column_stack((z,b,c,d)),train['y'],1.4,1e-4)
    basis = np.linalg.svd(train['traj'],full_matrices=False)[2][:2]
    sp = _kernel_predict(z,train['traj']@basis.T,.85,1e-4)@basis
    return {'y_hat':yh,'state_pred':sp,'diagnostics_vec':np.array([yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()])}
