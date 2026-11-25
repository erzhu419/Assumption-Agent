import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter
import pickle

def basis(t,e,tau,kind):
    t,e=np.asarray(t),np.asarray(e)
    dt, de=np.diff(t),np.diff(e)
    z=dt[:,None]/tau
    q=np.exp(-z)
    if kind=='ramp': u=de[:,None]*(-np.expm1(-z)/z)
    elif kind=='right': u=np.broadcast_to(de[:,None],z.shape)
    elif kind=='left': u=de[:,None]*q
    elif kind=='mid': u=de[:,None]*np.exp(-z/2)
    elif kind=='trap':
        v=np.gradient(e,t)
        u=dt[:,None]*(q*v[:-1,None]+v[1:,None])/2
    elif kind=='gradient':
        v=np.gradient(e,t)
        u=dt[:,None]*v[1:,None]*np.ones(len(tau))
    h=np.zeros((len(t),len(tau)))
    if np.allclose(dt,dt[0],rtol=1e-11,atol=0):
        for j in range(len(tau)): h[1:,j]=lfilter([1.],[1.,-q[0,j]],u[:,j])
    else:
        for j in range(1,len(t)): h[j]=q[j-1]*h[j-1]+u[j-1]
    return np.column_stack((e,h))

def solve(a,d,y):
    w=np.ones(len(y))
    for i in range(30):
        aw=a*w[:,None]; dw=d*w[:,None]; yw=y*w
        q,_=np.linalg.qr(dw,mode='reduced')
        yp=yw-q@(q.T@yw); ap=aw-q@(q.T@aw)
        c=nnls(ap,yp,maxiter=3000)[0]
        b=np.linalg.lstsq(dw,yw-aw@c,rcond=None)[0]
        r=y-a@c-d@b
        s=max(1e-10,1.4826*np.median(abs(r-np.median(r))))
        wn=(1+(r/(3*s))**2)**(-.5)
        if np.max(abs(wn-w))<1e-5: break
        w=wn
    return c,b,r,s

if __name__=='__main__':
    p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); x=p['experiment']; y=x['train_stress_raw']; t=x['train_t']; d=np.column_stack((np.ones_like(t),t/t[-1]-.5)); z=p['gt_test_stress']; gt=p['gt_G_inf']; tau=np.geomspace(.02,20,100)
    for kind in ['ramp','right','left','mid','trap','gradient']:
        a=basis(t,x['train_strain'],tau,kind); a2=basis(x['test_t'],x['test_strain'],tau,kind)
        c,b,r,s=solve(a,d,y); pred=a2@c
        # Public clean trace is a separate diagnostic of interpolation, never a fit input.
        cg=nnls(a2[:,1:],z-gt*a2[:,0],maxiter=3000)[0]; floor=np.linalg.norm(a2[:,1:]@cg+gt*a2[:,0]-z)/np.linalg.norm(z)
        print(kind,'trainMAD',round(s,5),'trainRMS98',round(np.sqrt(np.mean(np.sort(r*r)[:int(.98*len(r))])),5),'testRRMSE',round(np.linalg.norm(pred-z)/np.linalg.norm(z),6),'Ginf',round(c[0],4),'cleanBasisFloor',round(floor,8),'drift',b,'active',list(zip(np.round(tau[c[1:]>1e-6],4),np.round(c[1:][c[1:]>1e-6],3))))
    print('60-sample strain checkpoints',x['train_strain'][::60]); print('train largest strain increments',np.sort(abs(np.diff(x['train_strain'])))[-5:])
