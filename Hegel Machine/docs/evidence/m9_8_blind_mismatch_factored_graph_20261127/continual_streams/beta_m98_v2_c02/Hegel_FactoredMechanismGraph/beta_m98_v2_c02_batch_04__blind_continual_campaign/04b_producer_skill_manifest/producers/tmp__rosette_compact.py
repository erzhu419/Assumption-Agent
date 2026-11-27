from rosette_probe import x,y,design,fit
import numpy as np

def compact(z, j):
 return np.c_[np.ones(len(z)),z,z[:,j]*z[:,(j+1)%4],z[:,j]**3]

def model(z,v):
 a=[compact(z,j) for j in range(4)]; w=np.ones(len(z))
 for _ in range(25):
  sw=np.sqrt(w); b=np.array([np.linalg.lstsq(t*sw[:,None],v[:,j]*sw,rcond=None)[0] for j,t in enumerate(a)])
  r=np.linalg.norm(v-np.column_stack([t@b[j] for j,t in enumerate(a)]),axis=1)
  w=1/(1+(r/max(1e-7,2*np.median(r)))**2)
 keep=r<=max(1e-6,8*np.median(r)); return np.array([np.linalg.lstsq(t[keep],v[keep,j],rcond=None)[0] for j,t in enumerate(a)]),keep

def pred(z,b): return np.column_stack([compact(z,j)@b[j] for j in range(4)])
b,keep=model(x,y); np.set_printoptions(precision=10,suppress=True); print('coefficients',b,'excluded',np.where(~keep)[0]); print('clean max residual',np.max(abs(y-pred(x,b))[keep])); print('rank', [np.linalg.matrix_rank(compact(x[keep],j)) for j in range(4)])
radius=np.linalg.norm(x,axis=1); center=np.max(np.abs(x),axis=1)<.85; edge=radius>=np.quantile(radius,.75); rng=np.random.default_rng(62); scores=[]
for repeat in range(12):
 fold=rng.permutation(len(x))%8; yp=np.empty_like(y)
 for k in range(8):
  t=fold!=k; yp[~t]=pred(x[~t],model(x[t],y[t])[0]);
 r=abs(yp-y); scores.append([np.max(np.sqrt(np.mean(r[m&keep]**2,axis=0))) for m in (np.ones(len(x),bool),center,edge)])
print('worst repeated 8-fold RMSE: overall, center, edge',np.max(scores,axis=0)); print('forward slope singular values near sweep edge',np.linalg.svd(b[:,1:5],compute_uv=False)); print('fraction of all-reference span covered by predicted operating region')
