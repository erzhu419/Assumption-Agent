import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_batch_02_solver as base
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; original=base._axis
Y=np.array([base._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); W=np.array([p['w_pulse'] for p in pulses]); Z=s['Z_nuisance']; P=np.linalg.lstsq(Z,np.eye(len(Z)),rcond=None)[0]
def pred(par,model):
    count=len(par)//2
    def axis(k,v,h,robin):
        p=par[:count] if len(k)==s['nx'] else par[count:]
        if model=='wall':
            A=original(k,v,h,0).tolil()
            A[0,0]+=p[0]*robin/h; A[-1,-1]+=p[1]*robin/h
            if count==4:
                for i,j,f in [(0,1,p[2]),(-1,-2,p[3])]:
                    A[i,i]+=(f-1)*k[i]/h**2; A[i,j]-=(f-1)*k[i]/h**2
            return A.tocsc()
        if model=='interior':
            return original(k*p[0],v*p[1],h,p[2]*robin)
        if model=='wrap':
            A=original(k,v,h,0).tolil()
            for i,j in [(0,-1),(-1,0)]:
                t=p[0]*(k[0]+k[-1])/(2*h*h); A[i,i]+=t+p[1]*robin/h; A[i,j]-=t
            return A.tocsc()
    base._axis=axis
    try: S,_=base._design(s)
    finally: base._axis=original
    return W@S.T
R=pred([0,0,0,0],'wall')-Y
print('reflecting calibrated residual by pulse and shift',np.sqrt(np.mean(R.reshape(2,3,-1)**2,axis=2)))
print('signed means',R.reshape(2,3,-1).mean(2),'nuisance-projected rms',np.sqrt(np.mean((R-(R@P.T)@Z.T)**2,axis=1)))
for model,init,lo,hi in [('wall',[0]*4,-.3,3),('wall',[0,0,1,1]*2,-.3,3),('wrap',[0]*4,-.3,3),('interior',[1,1,0]*2,[-.1,-4,-.3]*2,[4,4,3]*2)]:
 for train in ([0],[1],[0,1]):
    def fun(q): return (pred(q,model)-Y)[train].ravel()
    fit=least_squares(fun,init,bounds=(lo,hi),max_nfev=130,gtol=1e-12)
    print(model,len(init),'train',train,'p',fit.x.round(5),'pulse rmse',np.sqrt(np.mean((pred(fit.x,model)-Y)**2,axis=1)))
