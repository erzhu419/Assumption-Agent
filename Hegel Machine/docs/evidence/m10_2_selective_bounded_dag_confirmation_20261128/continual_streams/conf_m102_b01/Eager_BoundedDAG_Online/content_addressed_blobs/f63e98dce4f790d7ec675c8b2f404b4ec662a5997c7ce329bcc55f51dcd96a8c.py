import numpy as np
from scipy.optimize import minimize_scalar


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-9)[0]


def _panel(u,v,features,bounds):
    def loss(k):
        a=features(u,k)
        return np.mean((a@_fit(a,v)-v)**2)
    k=minimize_scalar(loss,bounds=bounds,method='bounded').x
    q=_fit(features(u,k),v)
    return lambda x: features(x,k)@q


def _cos(x,k):
    return _aff(np.cos(k*x))


def _c1(x,k):
    return np.column_stack((_aff(x),np.arcsinh(k*x)))


def _c2(x,k):
    return np.column_stack((_aff(x),np.cbrt(1+(k*x)**2)))


import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

def aff(x): return np.c_[np.ones(len(x)),x]
def ls(a,y): return np.linalg.lstsq(a,y,rcond=1e-8)[0]
def quad(x):
    i,j=np.triu_indices(x.shape[1]); return x[:,i]*x[:,j]
def transform(t,k):
    if k=='a': return np.arctan(t)
    if k=='s': return np.arcsinh(t)
    if k=='t': return np.tanh(t)
    if k=='i': return np.sinh(np.arcsinh(t)/3)
    return t/np.sqrt(1+t*t)

def designs(z,c1,c2):
    yield 'C',aff(np.c_[z,c1,c2])
    for k in ['a','s','t','i','r']:
        for g in [1.1,1.8,2.6,3.4]:
            v=transform(g*z,k); q=quad(v)
            yield f'{k}/{g}/q',aff(np.c_[v,q]); yield f'{k}/{g}/c',aff(np.c_[v,q,c1,c2]); yield f'{k}/{g}/c2',aff(np.c_[v,c1,c2,quad(c1)]); yield f'{k}/{g}/d',aff(np.c_[v,q,c2]); yield f'{k}/{g}/b',aff(np.c_[v,q,c1]); yield f'{k}/{g}/j',aff(np.c_[v,q,np.sin(2.6*z),np.cos(2.6*z)])
    for name,v in [('c1',c1),('c2',c2),('z',z)]:
        for k in ['a','t','s']:
            for g in [0.8,1.6,2.6]:
                f=transform(g*v,k); yield f'{name}/{k}/{g}',aff(np.c_[f,quad(f),z]); yield f'{name}/{k}/{g}/f',aff(np.c_[f,quad(f),z,c1,c2])

def ridge(a,y,reg):
    mu=a[:,1:].mean(0); scale=np.maximum(a[:,1:].std(0),1e-6); d=(a[:,1:]-mu)/scale
    coef=np.linalg.solve(d.T@d+reg*np.eye(d.shape[1]),d.T@(y-y.mean()))/scale
    return np.r_[y.mean()-mu@coef,coef]

def proj(z,c1,c2,y,kind,base,mode):
    n=len(y); ii=np.arange(n); a=dict((k,x) for k,x in [('C',aff(np.c_[z,c1,c2])),('q',aff(np.c_[np.arctan(2.15*z),quad(np.arctan(2.15*z))])),('p',aff(np.c_[z,quad(z)])),('v',aff(np.c_[z,c1,c2,quad(np.arctan(1.8*z))]))])[base]; h={'z':z,'b':c1,'c':c2,'1':z[:,1:2],'i':np.arctan(1.8*z)}[mode]; mu=h[ii].mean(0); sd=np.maximum(h[ii].std(0),1e-6); h=aff((h-mu)/sd)
    f={'sin':np.sin,'cos':np.cos,'gauss':lambda x:np.exp(-x*x),'tanh':np.tanh,'cube':lambda x:x**3,'r':lambda x:x/(1+x*x),'erf':erf,'soft':lambda x:np.sqrt(1+x*x)}[kind]
    def des(p):return np.c_[a,f(h@p)]
    def fun(p):
        d=des(p)[:n]; return np.r_[d@ls(d,y)-y, .0004*p]
    raw=ls(h[:n],y); raw/=max(np.linalg.norm(raw[1:]),.05)
    target=y-a[:n]@ls(a[:n],y); r=ls(h[:n],target); r/=max(np.linalg.norm(r[1:]),.02)
    rng=np.random.default_rng(139); starts=[raw*g+np.r_[shift,np.zeros(h.shape[1]-1)] for g in [.35,1.,2.2] for shift in [0.,.75,-.75]]+[r,.5*r]+[rng.normal(0,.6,h.shape[1]) for k in range(4)]
    best=min((least_squares(fun,p,max_nfev=60,bounds=(-35,35)) for p in starts),key=lambda p:p.cost); d=des(best.x); return d[n:]@ls(d[:n],y)

def nonlinear(z,c1,c2,y,mode,kind,rank):
    h={'z':z,'c':c1,'d':c2,'a':np.arctan(1.6*z),'u':np.c_[z,c1,c2]}[mode]; n=len(y); h=aff((h-h[:n].mean(0))/np.maximum(h[:n].std(0),1e-6)); m=h.shape[1]; a=aff(np.c_[z,c1,c2]) if mode=='u' else aff(z); f={'t':np.tanh,'a':np.arctan,'s':np.sin,'r':lambda x:x/np.sqrt(1+x*x),'g':lambda x:np.exp(-x*x)}[kind];
    def design(p): return np.c_[a,f(h@p.reshape(m,rank))]
    def loss(p):
        d=design(p)[:n]; return np.r_[d@ridge(d,y,.0003)-y,.0015*p]
    rng=np.random.default_rng(14); r=ls(h[:n],y); r/=max(np.linalg.norm(r[1:]),.05)
    starts=[]
    for k in range(12):
        p=rng.normal(0,[.35,.8,1.6][k%3],(m,rank)); p[:,0]=r*[.5,1.4,2.5][k%3]; p[0,:]+=rng.uniform(-1,1,rank); starts.append(p.ravel())
    best=min((least_squares(loss,p,max_nfev=85,bounds=(-40,40)) for p in starts),key=lambda p:p.cost); d=design(best.x); return d[n:]@ridge(d[:n],y,.0003)



def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; w=np.asarray(inputs['node_00']['w']); b=s['panel_b']; c=s['panel_c']; tr=s['train']; n=len(tr['y'])
    a=_fit(_aff(b['u']),b['v']); u=(w-a[0])@np.linalg.pinv(a[1:]); z=_panel(b['u'],b['v'],_cos,(1.5,3.7))(u)
    c1=_panel(c['u'],c['v1'],_c1,(.2,4))(z)
    c2=_panel(c['u'],c['v2'],_c2,(.2,2))(z)
    yh=[]
    for spec in [('lin', 'z/a/2.6/f', 0.0003), ('lin', 's/3.4/j', 0.025), ('proj', 'soft', 'v', 'i')]:
        if spec[0]=='lin':
            d=dict(designs(z,c1,c2))[spec[1]]; yh.append(d[n:]@ridge(d[:n],tr['y'],spec[2]))
        else:
            fn=proj if spec[0]=='proj' else nonlinear
            yh.append(fn(z,c1,c2,tr['y'],*spec[1:]))
    y=np.mean(yh,axis=0)
    p=_c1(z,1.85); state=p[n:]@_fit(p[:n],tr['traj'])
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
