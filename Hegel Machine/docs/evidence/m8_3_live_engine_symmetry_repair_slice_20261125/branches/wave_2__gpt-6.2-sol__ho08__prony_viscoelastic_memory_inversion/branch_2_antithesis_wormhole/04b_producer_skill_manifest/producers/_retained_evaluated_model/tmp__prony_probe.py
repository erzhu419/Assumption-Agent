import pickle
import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter
from scipy.linalg import qr

def memory(t,e,tau,mode='linear'):
    dt=np.diff(t); z=dt[:,None]/tau; decay=np.exp(-z); fac={'linear':lambda:-np.expm1(-z)/z,'right':lambda:np.ones_like(z),'left':lambda:decay,'mid':lambda:np.exp(-z/2),'trap':lambda:(1+decay)/2}[mode]()
    drive=np.diff(e)[:,None]*fac
    if np.allclose(dt,dt[0],rtol=1e-11,atol=0):
        return np.vstack((np.zeros(len(tau)),np.column_stack([lfilter([1],[1,-a],u) for a,u in zip(decay[0],drive.T)])))
    out=np.zeros((len(t),len(tau)))
    for i in range(len(dt)): out[i+1]=decay[i]*out[i]+drive[i]
    return out

def fit(A,D,y,robust=True):
    w=np.ones(len(y))
    for it in range(20 if robust else 1):
        Aw=A*w[:,None]; Dw=D*w[:,None]; yw=y*w
        Q=qr(Dw,mode='economic')[0]
        Ap=Aw-Q@(Q.T@Aw); yp=yw-Q@(Q.T@yw)
        s=np.linalg.norm(Ap,axis=0); c=nnls(Ap/s,yp,maxiter=5000)[0]/s
        b=np.linalg.lstsq(Dw,(y-A@c)*w,rcond=None)[0]
        r=y-A@c-D@b
        mad=1.4826*np.median(np.abs(r-np.median(r)))
        w=1/np.hypot(1,r/max(mad*2.4,1e-10))
    return c,b,r,mad

if __name__=='__main__':
    p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; t,x,y,tt,xx=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]; yt=p['gt_test_stress']; D=np.column_stack((np.ones(len(t)),t/t[-1])); tau=np.geomspace(.02,20,100)
    print('strain slopes at integers',[(k,round(x[k*20],5),round((x[k*20]-x[k*20-1])*20,5)) for k in range(1,31)])
    for mode in ('linear','right','left','mid','trap'):
        A=np.column_stack((x,memory(t,x,tau,mode))); At=np.column_stack((xx,memory(tt,xx,tau,mode)))
        # Oracle diagnostic only: check which hereditary convention can represent the public clean waveform with the known equilibrium modulus.
        oracle=nnls(At[:,1:],yt-p['gt_G_inf']*xx,maxiter=5000)[0]; eo=np.linalg.norm(At[:,1:]@oracle+p['gt_G_inf']*xx-yt)/np.linalg.norm(yt)
        for robust in (False,True):
            c,b,r,mad=fit(A,D,y,robust); pred=At@c
            slices=[np.linalg.norm(pred[i]-yt[i])/np.linalg.norm(yt[i]) for i in np.array_split(np.arange(len(yt)),4)]
            print(mode,robust,'oracle %.7g'%eo,'G_inf %.5f'%c[0],'relRMSE %.6f'%(np.linalg.norm(pred-yt)/np.linalg.norm(yt)),'worst quartile %.6f'%max(slices),'mad %.5g'%mad,'drift',b,'modes',list(zip(tau[c[1:]>1e-6].round(4),c[1:][c[1:]>1e-6].round(4))))
