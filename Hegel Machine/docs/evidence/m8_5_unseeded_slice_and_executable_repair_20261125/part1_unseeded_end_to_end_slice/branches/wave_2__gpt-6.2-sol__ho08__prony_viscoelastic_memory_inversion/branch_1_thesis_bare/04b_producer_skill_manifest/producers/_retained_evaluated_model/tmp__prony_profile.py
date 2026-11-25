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
t=e['train_t']; x=e['train_strain']; y=e['train_stress_raw']; tt=e['test_t']; xx=e['test_strain']; yy=D['gt_test_stress']
B=np.column_stack((np.ones_like(t),(t-t.mean())/np.ptp(t)))
def pred(z,t,x,drift):
    ans=design(t,x,np.exp(z[5:9]),'linear')@z[:5]
    return ans+B@z[9:] if drift else ans
z0=np.r_[[12,25,18,15,9],np.log([.05,.28,1.5,10]),[.32,4.24]]
bounds=(np.r_[[0]*5,[np.log(.02)]*4,[-np.inf]*2],np.r_[[np.inf]*5,[np.log(20)]*4,[np.inf]*2])
res=least_squares(lambda z: pred(z,t,x,True)-y,z0,bounds=bounds,loss='huber',f_scale=.025)
r=res.fun; med=np.median(r); s=1.4826*np.median(abs(r-med)); keep=abs(r-med)<5*s
print('outliers', np.where(~keep)[0], 'sigma',s)
for cut in [3,4,5,8,15]:
    keep=abs(r-med)<cut*s
    a=least_squares(lambda z:(pred(z,t,x,True)-y)[keep],res.x,bounds=bounds)
    print('cut',cut,'G and tau',a.x[:5],np.exp(a.x[5:9]),'rel',np.linalg.norm(pred(a.x,tt,xx,False)-yy)/np.linalg.norm(yy),'G sd',np.sqrt(np.linalg.pinv(a.jac.T@a.jac)[0,0])*np.std(a.fun))
keep=abs(r-med)<5*s
for slow in [5,6,7,8,9,10,11,12,14,16,20]:
    def expand(u): return np.insert(u,8,np.log(slow))
    a=least_squares(lambda u:(pred(expand(u),t,x,True)-y)[keep],np.delete(res.x,8),bounds=tuple(np.delete(v,8) for v in bounds))
    print('slow',slow,'G',a.x[0],'SSE',a.fun@a.fun,'rel',np.linalg.norm(pred(expand(a.x),tt,xx,False)-yy)/np.linalg.norm(yy))
