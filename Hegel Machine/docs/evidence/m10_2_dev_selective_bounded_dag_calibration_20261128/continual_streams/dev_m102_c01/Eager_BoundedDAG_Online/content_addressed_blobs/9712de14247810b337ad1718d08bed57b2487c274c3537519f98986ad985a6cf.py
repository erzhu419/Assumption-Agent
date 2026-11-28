import numpy as np
from functools import lru_cache
from itertools import combinations_with_replacement
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


# Invert the known upstream Gram operator to recover the four input coordinates.
def recover(K,Kq):
 n=len(K); A=K-4; l,U=eigh(A); X=U[:,-4:]*np.sqrt(l[-4:]/2); j,k=np.triu_indices(n); ri=np.arange(len(j)); jj=np.arange(4)
 def value(a,b,w):
  d=a-b; d2=d*d; m=a*b; D=m.sum(1); e=np.exp(-d2.sum(1)/6); f=np.exp(-(d2@w)/6)
  pred=1+D+.15*(D*D+(m*m).sum(1))+1.5*(e+f)
  g=(1+.3*D[:,None])*b+.3*m*b-.5*d*(e[:,None]+f[:,None]*w)
  h=(1+.3*D[:,None])*a+.3*m*a+.5*d*(e[:,None]+f[:,None]*w)
  return pred,g,h,-.25*f[:,None]*d2
 def calc(v,jac):
  f=v[:-4].reshape(-1,4); P,G,H,W=value(f[j],f[k],v[-4:]);
  if not jac:return P-K[j,k]
  M=np.zeros((len(j),len(v))); M[ri[:,None],4*j[:,None]+jj]=G; M[ri[:,None],4*k[:,None]+jj]+=H; M[:,-4:]=W; return M
 best=None
 for seed in range(3):
  Z=X if seed==0 else X@np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4)))[0]; fit=least_squares(lambda v:calc(v,False),np.r_[Z.ravel(),np.ones(4)*.85],jac=lambda v:calc(v,True),bounds=(np.r_[np.full(4*n,-np.inf),np.full(4,.2)],np.r_[np.full(4*n,np.inf),np.full(4,3.)]),max_nfev=80,gtol=1e-10)
  if best is None or fit.cost<best.cost:best=fit
  if fit.cost<1e-14: break
 F=best.x[:-4].reshape(-1,4); w=best.x[-4:]; Q=[]
 for row in Kq:
  v=F[np.argmax(row)]; obj=lambda z:value(z[None,:],F,w); Q.append(least_squares(lambda z:obj(z)[0]-row,v,jac=lambda z:obj(z)[1],max_nfev=40).x)
 return F,np.asarray(Q).reshape(-1, 4),best.cost

_MODELS = [({'1': 1.0, 't': 0.01, 'sy1': 1.0, 'sy2': 0.3, 'x1': 1.0, 'x2': 0.3, 'sy4': 0.0778308077374508, 'str0.07': 26.529347014949618}, 0.0016632483978845916, 0), ({'1': 1.0, 't': 0.01, 'sy1': 1.0, 'sy2': 0.3, 'x1': 1.0, 'x2': 0.3, 'syr0.07': 0.12794906865003158, 'str0.2': 4.709378442696107, 'x4': 0.013181735956890296}, 0.00020458703071359245, 0), ({'1': 1.0, 't': 0.01, 'sy1': 1.0, 'sy2': 0.3, 'x1': 1.0, 'x2': 0.3, 'str0.2': 0.6059653484396205, 'sy4': 0.01819998485438697}, 0.00020786994353357523, 0), ({'1': 1.0, 't': 0.01, 'K': 1.0, 'sy4': 1.1363307036572223, 'sy2': 1.0905838285876612, 'xr0.07': 0.743560981404312}, 0.006500046161796381, 3), ({'1': 1.0, 't': 0.01, 'x1': 1.0, 'x2': 0.3, 'str1.0': 0.01367870032334943, 'xr0.45': 0.7756528589873036, 'syr0.07': 0.38367338316608773}, 0.016168850963265962, 2)]


@lru_cache(maxsize=None)
def _powers(dim, degree):
    return np.array(list(combinations_with_replacement(range(dim), degree)))


def _terms(p, state, x, q):
    score = p['scores'][:, 0]
    query = (state @ p['modes'].T)[:, 0]
    norm = max(float(score.std()), 1e-6)
    n, m = len(x), len(q)
    terms = {'1': (np.ones((n, n)), np.ones((m, n))),
             'K': (p['K'], p['Kq']),
             't': (np.outer(score, score)/norm**2, np.outer(query, score)/norm**2)}
    required = set().union(*(weights for weights, _, _ in _MODELS))
    for name, train, test in [('x', x, q), ('sy', p['sy'], p['qy']),
                               ('st', p['st'][:, :4], p['qt'][:, :4])]:
        mu = train.mean(0)
        train, test = train - mu, test - mu
        for key in required:
            if not key.startswith(name):
                continue
            suffix = key[len(name):]
            if suffix.startswith('r'):
                scale = float(suffix[1:])
                terms[key] = tuple(np.exp(-scale*cdist(a, train, 'sqeuclidean')) for a in (train, test))
            else:
                inds = _powers(train.shape[1], int(suffix))
                f, g = (a[:, inds].prod(axis=2) for a in (train, test))
                terms[key] = (f @ f.T, g @ f.T)
    return terms


def _fit(h, hq, y, ridge, robust):
    lam = np.full(len(y), ridge)
    for it in range(4 if robust else 1):
        c = cho_factor(h + np.diag(lam))
        alpha = cho_solve(c, y)
        if not robust or it == 3:
            break
        precision = cho_solve(c, np.eye(len(y))).diagonal()
        error = np.abs(alpha/np.sqrt(precision))
        cutoff = max(robust*1.4826*np.median(error), 1e-6)
        lam = ridge + np.maximum((error/cutoff)**2-1, 0)/precision
    return hq @ alpha


def run_node(inputs: dict) -> dict:
    p = inputs['node_00']
    state = inputs['node_02']['state_pred']
    y = p['y_centered'] + p['y_mean']
    with (threadpool_limits(limits=1, user_api='blas') if threadpool_limits else nullcontext()):
        x, q, cost = recover(p['K'], p['Kq'])
        terms = _terms(p, state, x, q)
        predictions = []
        for weights, ridge, robust in _MODELS:
            h, hq = (sum(weight*terms[key][i] for key, weight in weights.items()) for i in (0, 1))
            predictions.append(_fit(h, hq, y, ridge, robust))
        return {'y_hat': np.mean(predictions, axis=0)}
