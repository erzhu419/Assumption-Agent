import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares


def axis(k, v, h, adv, face):
    f = ((k[:-1] + k[1:])*.5 if face=='arith' else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]))/h**2
    if adv in ('center','negative'):
        a = ((v[:-1]+v[1:])*.5)/h * (1 if adv=='center' else -1)
        p = np.maximum(a,0); m = np.minimum(a,0)
        lower, upper = -f-p, -f+m
        d = np.r_[f+p,0] + np.r_[0,f-m]
    elif adv=='row':
        a=v/h; lower=-f-np.maximum(a[1:],0); upper=-f+np.minimum(a[:-1],0)
        d=np.r_[-upper,0]+np.r_[0,-lower]
    elif adv=='rowface':
        a=(v[:-1]+v[1:])*.5/h;lower=-f-np.maximum(a,0);upper=-f+np.minimum(a,0)
        d=np.r_[-upper,0]+np.r_[0,-lower]
    else:
        lower=upper=-f;d=np.r_[f,0]+np.r_[0,f]
    return diags([lower,d,upper],[-1,0,1],format='csc')


def prepare(s, adv='center', face='arith', grid=0):
    nx,ny=s['nx'],s['ny']; h=np.array([s['Lx']/(nx+grid),s['Ly']/(ny+grid)])
    ax=axis(s['kappa_x'],s['vel_x'],h[0],adv,face);ay=axis(s['kappa_y'],s['vel_y'],h[1],adv,face)
    ident=eye(nx*ny,format='csc')
    base=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    B=np.array([s['source_modes'][i].ravel(order='F') for i in range(len(s['source_modes']))]).T
    i,j=np.asarray(s['sensor_ij']).T; idx=i+nx*j
    wall=np.zeros((nx,ny,4));wall[0,:,0]=1;wall[-1,:,1]=1;wall[:,0,2]=1;wall[:,-1,3]=1
    wall=wall.reshape(-1,4,order='F')
    def predict(boundary,rhs=B):
        A=base+diags(wall@boundary)
        sol=[splu(A+m*ident).solve(rhs) for m in s['mu_shifts']]
        return np.concatenate([u[idx] for u in sol]),sol[-1]
    return predict, B, h


def calibrate(s):
    c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette']; off,*lin=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]; lin=np.array(lin)
    return lambda yy: np.linalg.solve(lin.T,(yy.reshape(-1,4)-off).T).T.ravel()

if __name__=='__main__':
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];cal=calibrate(s)
    ps=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in ps],axis=1); Y=np.stack([cal(p['y_pulse_raw']) for p in ps],axis=1)
    Z=s['Z_nuisance']; q=np.linalg.qr(Z)[0]; nom=np.repeat([s['robin_x'],s['robin_y']],2)
    print('pulse calibrated range',Y.min(0),Y.max(0),'noise behavior',Y[:6].T,'Z first rows',Z[:4]);print('scipy sparse okay')
    results=[]
    for ad in ['center','row','rowface','negative','none']:
     for face in ['arith','harm']:
      for grid in [-1,0,1]:
       f,B,h=prepare(s,ad,face,grid); rhs=B@W; hh=np.repeat(h,2); kap=np.array([s['kappa_x'][0],s['kappa_x'][-1],s['kappa_y'][0],s['kappa_y'][-1]])
       fits=[]
       for mode in range(2):
        def residual(b):
         e=f(b,rhs)[0]-Y
         return ((e-q@(q.T@e)) if mode else e).ravel()
        candidates=[nom,nom/hh,nom*kap/hh,nom/hh/(1+nom*hh/(2*kap)),nom*kap/hh/(1+nom*hh/2),np.zeros(4)]
        norms=[np.linalg.norm(residual(a))/np.sqrt(Y.size) for a in candidates]
        for n in (2,4):
         exp=lambda b: np.repeat(b,2) if n==2 else b
         r=least_squares(lambda b:residual(exp(b)), (nom/hh)[::2] if n==2 else nom/hh,bounds=(0,np.inf),gtol=1e-12)
         fits.append((np.sqrt(np.mean(r.fun**2)),np.round(exp(r.x),5)))
        if mode==0: raw=norms
       results.append((fits[2][0],ad,face,grid,np.round(raw,6),fits))
    for r in sorted(results,key=lambda t:t[0])[:10]:print(r)
