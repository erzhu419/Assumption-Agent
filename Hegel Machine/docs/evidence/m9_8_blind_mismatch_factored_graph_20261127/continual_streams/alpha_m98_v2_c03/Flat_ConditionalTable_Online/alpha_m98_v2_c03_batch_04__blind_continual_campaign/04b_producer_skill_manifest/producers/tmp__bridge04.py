import numpy as np
from scipy.optimize import least_squares

def basis(z, degree):
    return np.column_stack([np.ones(len(z))] + [z**p for p in range(1, degree+1)])

def forward_fit(z, x, degree):
    A = basis(z, degree)
    b = np.linalg.lstsq(A, x, rcond=None)[0]
    b = least_squares(lambda b: (A @ b.reshape(-1,4) - x).ravel(), b.ravel(), loss='soft_l1', f_scale=.003).x.reshape(-1,4)
    e = np.linalg.norm(A @ b - x, axis=1)
    clean = e < max(.003, 8*np.median(e))
    return np.linalg.lstsq(A[clean], x[clean], rcond=None)[0], clean

def inverse(x, b):
    z = (x-b[0]) @ np.linalg.inv(b[1:5])
    for _ in range(15):
        err = basis(z,(len(b)-1)//4) @ b - x
        jac = np.broadcast_to(b[1:5], (len(z),4,4)).copy()
        for p in range(2,(len(b)-1)//4+1):
            jac += p*z[:,:,None]**(p-1) * b[1+4*(p-1):1+4*p]
        z -= np.linalg.solve(jac.transpose(0,2,1),err[:,:,None])[:,:,0]
    return z

if __name__ == '__main__':
    import pickle
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; z=c['true_channel_loads']; x=c['bridge_raw']; rng=np.random.default_rng(816); fold=np.empty(len(z),int); fold[rng.permutation(len(z))]=np.arange(len(z))%5
    for deg in range(1,5):
        b,keep=forward_fit(z,x,deg); print('degree',deg,'rejected',np.flatnonzero(~keep),'training forward RMS',np.sqrt(np.mean((basis(z[keep],deg)@b-x[keep])**2)), 'coef',np.round(b,4).tolist())
        e=[]; g=[]
        for k in range(5):
            h=fold==k; a,_=forward_fit(z[~h],x[~h],deg); clean=h&keep; e.extend(inverse(x[clean],a).sum(1)-z[clean].sum(1)); g.extend(z[clean].sum(1))
        e=np.array(e); print('clean CV RMS/max/worst decile RMS', np.sqrt(np.mean(e**2)), max(abs(e)), max(np.sqrt(np.mean(a**2)) for a in np.array_split(e[np.argsort(g)],10)))
