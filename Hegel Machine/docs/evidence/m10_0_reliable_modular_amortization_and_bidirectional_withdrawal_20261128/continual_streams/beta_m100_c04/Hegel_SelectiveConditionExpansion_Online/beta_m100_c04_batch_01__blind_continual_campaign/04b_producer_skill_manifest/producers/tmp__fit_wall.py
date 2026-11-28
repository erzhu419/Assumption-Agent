import numpy as np, pickle
from scipy import sparse
from scipy.optimize import least_squares
import chamber_baseline as base
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T; Y=np.array([base._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]).T
axis=base._axis; Z=s['Z_nuisance']; proj=np.eye(len(Y))-Z@np.linalg.pinv(Z)
# Diagonal wall loss and inward transport multiplier at each of x0,x1,y0,y1.
def forward(p):
    i=iter(p.reshape(4,2))
    def mod(k,v,h,r):
        A=axis(k,v,h,r).tolil()
        for j,l in ((0,1),(-1,-2)):
            d,m=next(i); a=k[j]/h**2*m
            A[j,j]=a+d*r/h; A[j,l]=-a
        return A.tocsc()
    base._axis=mod
    try: return base._forward(s)[0]@W
    finally: base._axis=axis

def test(label,p):
    D=forward(p)-Y
    print(label,'p=',np.round(p,7),'pulse rms',np.sqrt(np.mean(D**2,axis=0)),'projected',np.sqrt(np.mean((proj@D)**2,axis=0)),flush=True)

test('open',np.ones(8)); test('reflect',np.tile([0,1],4)); test('loss only',np.tile([1,0],4)); print('signal rms',np.sqrt(np.mean(Y**2,axis=0)))
for use_proj in [False,True]:
    P=proj if use_proj else np.eye(len(Y))
    for cols in [[0,1,2],[1,2],[0,2],[0,1]]:
        fit=least_squares(lambda p:(P@(forward(p)-Y))[:,cols].ravel(), np.ones(8), bounds=(0,15),ftol=1e-12,xtol=1e-12,gtol=1e-12)
        test(('fit',use_proj,cols),fit.x); print('jac sv',np.round(np.linalg.svd(fit.jac,compute_uv=False),6))
print('walls k and v',[(s['kappa_'+d][[0,1,-2,-1]],s['vel_'+d][[0,-1]]) for d in 'xy'])
