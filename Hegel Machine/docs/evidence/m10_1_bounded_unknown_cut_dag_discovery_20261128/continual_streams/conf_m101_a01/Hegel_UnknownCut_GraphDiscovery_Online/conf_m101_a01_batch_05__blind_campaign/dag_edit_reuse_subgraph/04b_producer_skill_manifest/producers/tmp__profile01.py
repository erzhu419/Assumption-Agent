exec(open('/tmp/inspect_node01.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
pairs=list(combinations_with_replacement(range(4),2))
def quad(x): return np.column_stack([x[:,i]*x[:,j] for i,j in pairs])
def affine(x): return F(x,x)
def base(x): return F(x,x,quad(x))
def design(x,p,mode):
    h=p[0]+x@p[1:5]; D=1+h*h
    if mode=='rat': return np.column_stack([affine(x),quad(x)/D[:,None]])
    if mode=='ratfull': return np.column_stack([affine(x),base(x)/D[:,None]])
    if mode=='rataxis': return np.column_stack([affine(x),quad(x),1/D])
    if mode=='gate': return np.column_stack([base(x),affine(x)*np.tanh(h)[:,None]])
    if mode=='tanh': return np.column_stack([base(x),np.tanh(h)])
    if mode=='sin': return np.column_stack([base(x),np.sin(h)])
    if mode=='cosgate': return np.column_stack([base(x),affine(x)*np.cos(h)[:,None]])
    if mode=='rat2': return np.column_stack([affine(x),quad(x)/D[:,None],quad(x)])
    if mode=='rank2rat':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j/D])
    if mode=='rank2sin':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j,np.sin(h)])
    if mode=='rank2gate':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j,np.tanh(h)])
    if mode=='rank2':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j])
ye=np.linalg.eigh(np.linalg.lstsq(quad(q),y-affine(q)@fit(affine(q),y),rcond=None)[0][:4,None]@np.ones((1,4))) if False else None

def train(x,y,mode,n=8):
    b=fit(base(x),y); M=np.zeros((4,4));
    for k,(i,j) in enumerate(pairs): M[i,j]=M[j,i]=b[k+5]/(1+(i!=j))
    e,V=np.linalg.eigh(M); a=V[:,-1]*np.sqrt(abs(e[-1])); b=V[:,0]*np.sqrt(abs(e[0])); g=a+b; j=a-b
    def residual(p):
        X=design(x,p,mode); return X@fit(X,y)-y
    rng=np.random.default_rng(495); best=None
    for k in range(n):
        pp=np.r_[0.,np.eye(4)[k%4]*(.7 if 'rat' in mode else 1.7)] if k<4 else rng.normal(size=5)*(.8 if 'rat' in mode else 1.7)
        if mode.startswith('rank2'): pp=np.r_[pp,g,j]
        f=least_squares(residual,pp,bounds=(-7,7),max_nfev=125,ftol=2e-7); v=np.mean(f.fun**2)
        if best is None or v<best[0]: best=(v,f.x)
    return best
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
for m in ('rank2','rank2rat','rank2sin','rank2gate','rat','ratfull','rataxis','gate','tanh','sin','cosgate','rat2'):
    es=[]
    for k,(t,v) in enumerate(splits):
        var,p=train(q[t],y[t],m); X=design(q,p,m); pred=X[v]@fit(X[t],y[t]); es.append(np.mean(abs(pred-y[v])))
        if k==0: print(m,'fit RMSE',var**.5,'p',p,'public',es[0],flush=True)
    print(m,'folds',np.round(es,5),'worst',max(es),flush=True)
