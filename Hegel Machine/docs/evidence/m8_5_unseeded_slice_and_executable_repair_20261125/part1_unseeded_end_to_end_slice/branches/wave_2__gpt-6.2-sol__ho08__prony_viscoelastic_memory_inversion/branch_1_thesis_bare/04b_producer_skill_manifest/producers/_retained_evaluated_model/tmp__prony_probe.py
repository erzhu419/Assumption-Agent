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

t=e['train_t']; x=e['train_strain']; y=e['train_stress_raw']; tt=e['test_t']; xx=e['test_strain']; yy=D['gt_test_stress']; tau=np.geomspace(.02,20,91)
for method in ['linear','right','left','mid','trap']:
    X=design(t,x,tau,method); beta,r,b=fit(X,t,y)
    keep=np.abs(r-np.median(r))<5*1.4826*np.median(np.abs(r-np.median(r))); beta2,r2,b2=fit(X[keep],t[keep],y[keep]);
    for c,rr,bb in [(beta,r,b),(beta2,r2,b2)]:
        Z=design(tt,xx,tau,method); p=Z@c; print(method,'rel',np.linalg.norm(p-yy)/np.linalg.norm(yy),'inf',c[0],'modes',list(zip(tau[c[1:]>0].round(4),c[1:][c[1:]>0].round(3))),'noise',1.4826*np.median(abs(rr-np.median(rr))),'drift',bb)
print('different training rates by time:');
for i in range(0,len(x)-1,20): print(t[i],x[i],np.round(np.diff(x[i:i+21])[:8]/.05,4))
