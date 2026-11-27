import contextlib, io, runpy
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','d','tr','te','di','dt','t','tau','theta','bridge','n','h','v','vr')})
x=m._features(di); past,hold,_=m._coupon(di,x,bridge); f=x@theta[3:]; O=np.c_[v,vr]; labs=di['lab_id']; W=np.r_[np.ones(len(t)),1.4]; L=np.r_[t,1.]; E=lambda a:np.r_[-np.expm1(-t/a),1/a]; S=sum(w*E(a) for w,a in zip(theta[:2],tau)); Y=O-hold[:,None]*S

def fit(rows, spec):
    form, offset, feat, memory = spec
    names=['linear','one','two','rational','fixed']; starts={'linear':[[]], 'one':[[1.1],[3.]], 'two':[[.3,1.4],[.6,3.]], 'rational':[[1.1],[3.]], 'fixed':[[]]}
    def design(z):
        a=np.exp(z); B={'linear':lambda:[L], 'one':lambda:[E(a[0]),L], 'two':lambda:[E(a[0]),E(a[1]),L], 'rational':lambda:[np.r_[t/(a[0]+t),1/a[0]], L], 'fixed':lambda:[E(tau[0]),E(tau[1]),L]}[form]()
        if feat=='free': D=[x[:,j,None]*b for b in B for j in range(4)]
        else: D=[(f+theta[2])[:,None]*b for b in B]
        if offset=='free': D.extend(np.broadcast_to(b,Y.shape) for b in B)
        if offset=='linear': D.append(np.broadcast_to(L,Y.shape))
        if memory: D.extend(past[:,j,None]*E(tau[0]) for j in range(past.shape[1]))
        return np.stack(D,axis=-1)
    def proj(z, pred=False):
        A=design(z); a=(A[rows]*W[None,:,None]).reshape(-1,A.shape[-1]); coef=np.linalg.lstsq(a,(Y[rows]*W).ravel(),rcond=1e-8)[0]; P=A@coef
        return (P,coef) if pred else ((P[rows]-Y[rows])*W).ravel()
    cand=[]
    for s in starts[form]:
        z=np.log(s)
        if len(z): z=least_squares(proj,z,bounds=(np.log(.07),np.log(15.)),max_nfev=65).x
        cand.append((np.mean(proj(z)**2),z))
    _,z=min(cand,key=lambda el:el[0]); P,c=proj(z,True); return P,c,np.exp(z)

specs=[(form,off,'linked',False) for form in ('linear','one','two','rational','fixed') for off in ('none','linear','free') if not(form=='linear' and off=='free')]; specs.extend([('two','linear','free',False),('two','linear','linked',True),('one','linear','linked',True)])
for spec in specs:
    P,c,a=fit(np.ones(len(Y),bool),spec); fold=[]
    for lab in np.unique(labs):
        rows=labs==lab; Q,_,_=fit(~rows,spec); fold.append(np.sqrt(np.mean((Q[rows]-Y[rows])**2,axis=0)))
    fold=np.asarray(fold); last=fold[:,-2]; print(spec,'fit',np.sqrt(np.mean((P-Y)**2,axis=0))[[0,-2,-1]].round(5),'worst-lab early/late/rate',fold[:,[0,-2,-1]].max(0).round(5),'mean lab',fold[:,[0,-2,-1]].mean(0).round(5),'tau',a.round(4),'coef',c.round(4))
for label,g in [('diag',di),('test',te)]:
    X=m._features(g) if label=='diag' else m._features(tr,te)[n:]; print(label,'normalized feature projection range',np.quantile(X@theta[3:],[0,.25,.5,.75,1]).round(4),'hold range',np.quantile(m._coupon(g,X,bridge)[1],[0,.5,1]).round(4))
