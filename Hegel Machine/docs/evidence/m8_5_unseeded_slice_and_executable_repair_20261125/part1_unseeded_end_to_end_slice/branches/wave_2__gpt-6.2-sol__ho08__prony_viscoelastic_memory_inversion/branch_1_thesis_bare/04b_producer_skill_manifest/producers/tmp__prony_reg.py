import pickle
import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter
D=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=D['experiment']
def design(t,x,tau,method):
    dt=t[1]-t[0]; h=dt/tau; decay=np.exp(-h)
    factor={'linear':-np.expm1(-h)/h,'right':np.ones_like(h),'left':decay,'mid':np.exp(-h/2),'trap':(1+decay)/2}[method]
    return np.column_stack([x,*[amp*lfilter([1],[1,-a],np.r_[0,np.diff(x)]) for a,amp in zip(decay,factor)]])

def fit(X,t,y):
    B=np.column_stack([np.ones_like(t),(t-t.mean())/np.ptp(t)])
    w=np.ones(len(y))
    for it in range(20):
        Xw=X*w[:,None]; Bw=B*w[:,None]; yw=y*w
        q=np.linalg.qr(Bw)[0]; Xp=Xw-q@(q.T@Xw); yp=yw-q@(q.T@yw)
        beta=nnls(Xp,yp,maxiter=3000)[0]; b=np.linalg.lstsq(Bw,(y-X@beta)*w,rcond=None)[0]; r=y-X@beta-B@b
        s=1.4826*np.median(np.abs(r-np.median(r))); w=np.sqrt(np.minimum(1,1.345*s/np.maximum(1e-14,abs(r))))
    return beta,r,b

from scipy.optimize import least_squares
t=e['train_t']; x=e['train_strain']; y=e['train_stress_raw']; tt=e['test_t']; xx=e['test_strain']; yy=D['gt_test_stress']; B=np.column_stack((np.ones_like(t),(t-t.mean())/np.ptp(t)))
tau=np.geomspace(.02,20,81); X=design(t,x,tau,'linear')
def refine(y, strengths):
    c,r,b=fit(X,t,y); med=np.median(r); scale=1.4826*np.median(abs(r-med)); keep=abs(r-med)<5*scale
    idx=np.flatnonzero(c[1:]>1e-6); groups=np.split(idx,np.flatnonzero(np.diff(np.log(tau[idx]))>np.log(1.8))+1)
    bands=np.array([np.average(np.log(tau[g]),weights=c[1:][g]) for g in groups]); m=len(groups)
    z0=np.r_[c[0],[np.sum(c[1:][g]) for g in groups],bands,b]; L=np.r_[[0.]*(m+1),[np.log(.02)]*m,[-np.inf]*2]; U=np.r_[[np.inf]*(m+1),[np.log(20)]*m,[np.inf]*2]; k=slice(m+1,2*m+1)
    def fun(z, strength):
        vals=design(t,x,np.exp(z[k]),'linear')@z[:m+1]+B@z[-2:]-y
        return np.r_[vals[keep],strength*scale*np.diff(z[k],n=2)]
    result=[]
    for s in strengths:
        a=least_squares(fun,np.clip(z0,L,U),args=(s,),bounds=(L,U),max_nfev=100)
        result.append((a.x[0],design(tt,xx,np.exp(a.x[k]),'linear')@a.x[:m+1], np.exp(a.x[k])))
    return result
strengths=[0, 2.8, 5.6, 8, 12, 20]
for s,(g,p,tm) in zip(strengths, refine(y,strengths)): print(s,g,np.linalg.norm(p-yy)/np.linalg.norm(yy),tm)
rng=np.random.default_rng(9825)
for scenario,taus in enumerate([[.045,.27,1.55,8.5],[.03,.22,1.5,12],[.08,.18,.9,5],[.04,.4,2.2,7],[.04,.18,1.2,4,14],[.06,.55,5],[.28,2.2,13]]):
    coeff=np.r_[12,rng.uniform(7,25,len(taus))]; tx=design(t,x,np.array(taus),'linear')@coeff+B@np.array([.3,4]); clean=design(tt,xx,np.array(taus),'linear')@coeff
    scores=[]
    for _ in range(12):
        obs=tx+rng.normal(0,.016,len(t)); idx=rng.choice(len(t),size=12,replace=False); obs[idx]+=rng.choice([-1,1],12)*rng.uniform(.3,5,12)
        scores.append([[abs(g-12)/12,np.linalg.norm(p-clean)/np.linalg.norm(clean)] for g,p,_ in refine(obs,strengths)])
    scores=np.array(scores); print(scenario,taus,'G med',np.round(np.median(scores[:,:,0],axis=0),3),'G pass',np.mean(scores[:,:,0]<=.06,axis=0),'test median',np.round(np.median(scores[:,:,1],axis=0),4))
