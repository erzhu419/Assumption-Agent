import numpy as np
from scipy.optimize import least_squares
def d(*x):return np.column_stack((np.ones(len(x[0])),*x))
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-11)[0]
def af(u,q):return d(u,u**3,np.sin(q[0]*u),np.tanh(q[1]*u))
def bf(u,q):return d(u*u,np.cos(q[0]*u),np.exp(-q[1]*u*u))
def cf(u,q):return d(u,u*u,np.cos(q[0]*u),np.exp(-q[1]*u*u))
def c1f(u):return d(u*np.roll(u,-1,axis=1))
def op(s,key,vkey,F,q0):
 u,v=s[key]['u'],s[key][vkey]; keep=np.arange(len(u)); f=F(u,q0)
 if key=='panel_a':
  for _ in range(8):
   err=np.linalg.norm(f@fit(f[keep],v[keep])-v,axis=1);keep=np.argsort(err)[:40]
 def res(q):
  f=F(u[keep],q);return (f@fit(f,v[keep])-v[keep]).ravel()
 for _ in range(3):
  q=least_squares(res,q0,bounds=([.3,.3],[5.,6.]),gtol=1e-12).x; f=F(u,q); w=fit(f[keep],v[keep]); err=np.linalg.norm(f@w-v,axis=1)
  if key=='panel_a': keep=np.flatnonzero(err<max(.001,4*np.median(err)))
 return lambda x:F(x,q)@w,w

from itertools import combinations_with_replacement

def coordinates(s,x):
 A,aw=op(s,'panel_a','v',af,[2.15,2.15]); B,_=op(s,'panel_b','v',bf,[2.15,2.15]); C2,_=op(s,'panel_c','v2',cf,[2.15,2.15]); pc=s['panel_c']; cw=fit(c1f(pc['u']),pc['v1']); z0=np.linalg.solve(aw[1:5].T,(x-aw[0]).T).T
 z=np.array([least_squares(lambda z:A(z[None])[0]-xi,np.clip(zi,-1.3,1.3),bounds=(-2,2),gtol=1e-12).x for xi,zi in zip(x,z0)])
 return dict(z=z,b=B(z),c=c1f(z)@cw,h=C2(z))

def quad(x):
 return np.stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)],axis=1)

def temporal(s):
 t=np.asarray(s['eval_times']); T=np.asarray(s['train']['traj']); F=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(F(p)@fit(F(p),T.T)-T.T).ravel(),[.42,1.81],bounds=([.10,.7],[.7,4.]),gtol=1e-12).x; f=F(p)
 return f,fit(f,T.T).T

def residpred(F,K,y):
 n=len(y); q=np.linalg.pinv(F[:n],rcond=1e-9); v=q@y; R=np.eye(n)-F[:n]@q
 P=K[:, :n]-F@(q@K[:n,:n]); return F@v+P@R@np.linalg.solve(R@K[:n,:n]@R+.1*np.eye(n),R@y)

def baseline(study: dict) -> dict:
 tr=study['train']; n=len(tr['x']); cc=coordinates(study,np.r_[tr['x'],study['test']['x']]); z,b,c,h=[cc[k] for k in 'zbch']; f,m=temporal(study); qb=quad(b)
 K=np.zeros((len(z),len(z)))
 for X,w in [(z,.6),(b,.35),(c,.15),(h,.15)]:
  X=X/np.maximum(X[:n].std(0),.1); D=((X[:,None]-X[None,:])**2).sum(-1); K+=w*(np.exp(-D/8)+np.exp(-D/2))
 pred=np.column_stack([residpred(d(z),K,m[:,0]),residpred(d(qb),K,m[:,1])]); y0=d(z[:n],qb[:n],m); y=d(z,qb,pred)[n:]@fit(y0,tr['y']); state=pred[n:]@f.T; end=state[:,-1]
 return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}

from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _blocks(study,z,b,c,h):
    n=len(study['train']['x']); q=np.asarray(study['telemetry_vec']); out={}
    def put(k,a):
        a=np.asarray(a).reshape(len(z),-1); a=a-a[:n].mean(0); out[k]=a/np.maximum(a[:n].std(0),.035)
    def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
    for k,a in zip('zbch',(z,b,c,h)):
        put(k,a); put(k+'2',quad(a)); put(k+'3',a**3); put(k+'4',a**4)
    for k,a,j,g in [('z',z,'b',b),('z',z,'c',c),('z',z,'h',h),('b',b,'c',c),('b',b,'h',h),('c',c,'h',h)]:
        put(k+j,prod(a,g)); put(k+j+'2',prod(a,quad(g)))
    for k,a in zip('zbch',(z,b,c,h)):
        put(k+'s',np.c_[np.sin(q[1]*a),np.cos(q[1]*a),np.tanh(q[0]*a),a/(1+q[0]*a*a)])
        for m,r in [('z',z),('b',b)]:
            put(m+k+'gate',prod(r,1/(1+q[2]*a*a))); put(m+k+'sat',prod(r,np.tanh(q[0]*a)))
    put('bqz',prod(quad(b),z)); put('bqh',prod(quad(b),h)); put('bqc',prod(quad(b),c))
    put('bqzg',prod(quad(b),1/(1+q[2]*z*z)));put('bqhg',prod(quad(b),1/(1+q[2]*h*h)))
    put('bq',quad(b)[:,[1,3,5,8]])
    B,_=op(study,'panel_b','v',bf,[2.15,2.15]); C,_=op(study,'panel_c','v2',cf,[2.15,2.15]); pc=study['panel_c']; cw=fit(c1f(pc['u']),pc['v1'])
    for k,a in enumerate((np.tanh(q[0]*z),np.sin(q[1]*z),z/(1+q[2]*z*z),b,h,c)):
        bb=B(a); put(f'balt{k}',np.c_[bb,quad(bb)]); put(f'calt{k}',np.c_[C(a),c1f(a)@cw])
    return out

def _kernels(blocks):
    K={}
    for k,a in blocks.items():
        K[k]=(a@a.T)/a.shape[1]
    for k in 'zbch':
        a=blocks[k]; dist=cdist(a,a,'sqeuclidean'); K[k+'rb']=.5*(np.exp(-dist/4)+np.exp(-dist/14))
    for nm in ('zb','zh','zbh','bc','bh','bzch'):
        D=cdist(np.column_stack([blocks[k]/np.sqrt(len(nm)) for k in nm]),np.column_stack([blocks[k]/np.sqrt(len(nm)) for k in nm]),'sqeuclidean');K[nm+'rb']=.5*(np.exp(-D/4)+np.exp(-D/14))
    return K

def _head(F,y,K,groups):
    n=len(y); Gram=np.stack([sum(K[k] for k in g)/len(g) for g in groups]); pre=np.ascontiguousarray(Gram[:,:n,:n]).reshape(len(groups),-1); cross=np.ascontiguousarray(Gram[:,n:,:n]).reshape(len(groups),-1)
    Q=np.linalg.qr(F[:n],mode='complete')[0][:,F.shape[1]:]; v=Q.T@y; r=len(v); vv=max(np.mean(v*v),1e-6); blocks=np.stack([Q.T@k[:n,:n]@Q for k in Gram]); flat=blocks.reshape(len(groups),-1); I=np.eye(r)
    def objective(x):
        w=np.exp(x); C=(w[:-1]@flat).reshape(r,r)+w[-1]*I; cf=cho_factor(C,check_finite=False); a=cho_solve(cf,v,check_finite=False); J=cho_solve(cf,I,check_finite=False)-np.outer(a,a)
        loss=.5*v@a+np.log(np.diag(cf[0])).sum()+.025*np.sum((x[:-1]-np.log(vv/max(len(groups),1)))**2)
        grad=.5*w*np.r_[flat@J.ravel(),np.trace(J)]; grad[:-1]+=.05*(x[:-1]-np.log(vv/max(len(groups),1))); return loss,grad
    x0=np.log(np.r_[np.full(len(groups),vv*.3/len(groups)),vv*.7]); x=minimize(objective,x0,jac=True,method='L-BFGS-B',bounds=[(-17,4)]*len(x0),options={'maxiter':110}).x; w=np.exp(x)
    U=(w[:-1]@pre).reshape(n,n)+w[-1]*np.eye(n); a=Q@cho_solve(cho_factor(Q.T@U@Q),v); coef=fit(F[:n],y-U@a); return F[n:]@coef+(w[:-1]@cross).reshape(len(F)-n,n)@a

def _new(study,variant=0):
    tr=study['train']; n=len(tr['x']); cc=coordinates(study,np.r_[tr['x'],study['test']['x']]); z,b,c,h=[cc[k] for k in 'zbch']; f,m=temporal(study); qb=quad(b); blocks=_blocks(study,z,b,c,h); K=_kernels(blocks)
    common=[('zrb',),('brb',),('crb','hrb'),('zbrb','zhrb','zbhrb','bcrb','bhrb','bzchrb'),('zs','bs','cs','hs'),('zc','zh','bc','bh','ch'),('zzgate','zbgate','zcgate','zhgate','bbgate','bcgate','bhgate'),('balt0','balt1','balt2','calt0','calt1','calt2'),('balt3','balt4','balt5','calt3','calt4','calt5')]
    G0=[('z2',),('z3',),('b',),('c',),('h',),('bq',),('b2',),('zb',),('zb2',),('zzsat','zbsat','zcsat','zhsat'),('z4','h2','c2')]+common
    G1=[('z',),('b',),('c',),('h',),('z2','z3'),('b3',),('b4',),('bqzg','bqhg'),('bqz','bqc','bqh'),('zb',),('bc2','bh2','bcsat','bhsat')]+common
    if variant==1:
        G0=G0[:9]+common[:5]; G1=G1[:8]+common[:5]
    F0=d(z); F1=d(qb)
    if variant==2:
        F1=d(qb[:,[1,3,5,8]]); G1=[('b2',)]+G1
    pred=np.column_stack((_head(F0,m[:,0],K,G0),_head(F1,m[:,1],K,G1))); sel=[1,3,5,8]; y=d(z[n:],qb[n:][:,sel],pred)@fit(d(z[:n],qb[:n][:,sel],m),tr['y']); state=pred@f.T
    return _out(y,state)

def _out(y,state):
    end=state[:,-1]; return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}

_VARIANT=2
_BLEND=1

def solve_batch(study:dict)->dict:
    if _BLEND==0:return baseline(study)
    p=_new(study,_VARIANT)
    if _BLEND==1:return p
    q=baseline(study); return _out(_BLEND*p['y_hat']+(1-_BLEND)*q['y_hat'], _BLEND*p['state_pred']+(1-_BLEND)*q['state_pred'])
