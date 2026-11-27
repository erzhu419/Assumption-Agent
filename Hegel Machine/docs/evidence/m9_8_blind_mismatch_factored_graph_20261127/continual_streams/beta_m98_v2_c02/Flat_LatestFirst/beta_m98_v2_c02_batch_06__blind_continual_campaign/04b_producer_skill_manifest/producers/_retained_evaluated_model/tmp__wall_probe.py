from cal_probe import item,s,c,x,y,solo,load,robust,bases
import numpy as np
from itertools import product
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import root

m = load('batch_04')
coef=m._fit_rosette(c)
lin = (m._rosette_raw(np.eye(4), coef * [1,1,1,1,0,0]) - coef[:,0]).T

def invert(obs):
    raw=np.asarray(obs).reshape(-1,4)
    return np.array([root(lambda a: m._rosette_raw(a,coef)-b,np.linalg.solve(lin,b-coef[:,0]),tol=1e-11).x for b in raw]).ravel()

pulses=item['campaign_diagnostics']; cal=[invert(p['y_pulse_raw']) for p in pulses]
Z=s['Z_nuisance']; project=lambda r:r-Z@np.linalg.lstsq(Z,r,rcond=None)[0]
nx,ny=s['nx'],s['ny']; print('geometry and shifts',nx,ny,s['mu_shifts'],'robin',s['robin_x'],s['robin_y']); print('public study keys',list(s))
sources=s['source_modes'].reshape(-1,nx*ny,order='F').T
ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; ident=sparse.eye(nx*ny,format='csc')
reac=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); periodic=load('batch_03')._axis

def forward(kx,ky):
    axes=[]
    for a,kind,n in [('x',kx,nx),('y',ky,ny)]:
        k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(n-1)
        axes.append(periodic(k,v,h) if kind=='periodic' else m._axis(k,v,h,s['robin_'+a] if kind=='open' else 0.))
    ax,ay=axes
    L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+reac
    return np.vstack([splu(L+mu*ident).solve(sources)[rows] for mu in s['mu_shifts']])

def slices(r):
    return np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1)**2,axis=1)).round(7).tolist()

for kx,ky in product(('open','sealed','periodic'),repeat=2):
    S=forward(kx,ky); stats=[]
    for p,obs in zip(pulses,cal):
        pred=S@p['w_pulse']; raw=m._rosette_raw(pred.reshape(-1,4),coef).ravel()
        r=pred-obs
        stats.append((slices(raw-p['y_pulse_raw']),slices(r),slices(project(r))))
    print(kx,ky,'raw/calibrated/profiled shift RMSE by pulse:',stats)

for j in range(4):
    terms='1 a b c ac a3'.split(); A=np.column_stack([bases(x,j)[k] for k in terms]); F=np.c_[np.ones(len(x)),x]; ef=[]; ec=[]
    mask=solo['bank_id']==j; yb=solo['raw_readout'][mask]; xb=np.eye(4)[j]*solo['ref_signal'][mask,None]; B=np.column_stack([bases(xb,j)[k] for k in terms]); Fb=np.c_[np.ones(len(xb)),xb]; eb=[]
    for fold in range(5):
        test=np.arange(len(x))%5==fold
        q,r=robust(A[~test],y[~test,j]); good=abs(r)<1e-3
        af=np.linalg.lstsq(F[~test][good],y[~test,j][good],rcond=None)[0]
        ef.extend((F[test]@af-y[test,j]).tolist()); ec.extend((A[test]@q-y[test,j]).tolist()); eb.append(np.abs(B@q-yb))
    print('bank',j,'held-out affine/cubic median abs',np.median(abs(np.array(ef))),np.median(abs(np.array(ec))),'held-out cubic p90',np.quantile(abs(np.array(ec)),.9),'solo median abs',np.median(eb))
