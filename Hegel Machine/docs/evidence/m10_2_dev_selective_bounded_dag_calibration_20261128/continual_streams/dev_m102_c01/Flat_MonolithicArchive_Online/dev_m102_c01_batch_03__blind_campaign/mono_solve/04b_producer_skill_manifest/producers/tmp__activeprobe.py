import sys,contextlib,io
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_fit
with contextlib.redirect_stdout(io.StringIO()): from pathprobe import *
for i,j in splits:
 ss=dict(s,train={k:v[i] if k!='summary' else v for k,v in s['train'].items()},test={'x':x[j]}); p=solve_batch(ss); print('saved',np.mean(abs(p['y_hat']-y[j])),np.mean(abs(p['state_pred']-t[j])))

# Structured stationarity and additive low-order interactions in panel coordinates.
def bases(q):
 d=q.shape[1]; L=q@q.T; E=cdist(q,q,'sqeuclidean'); yield 'rbf',L,E
 for flip in (0,1):
  A=np.eye(d); ij=np.triu_indices(d,1); A=np.c_[A,(A[:,ij[0]]+(-1 if flip else 1)*A[:,ij[1]])/np.sqrt(2)]; f=q@A; G=(f[:,None,:]-f[None,:,:])**2
  for ell in (.15,.35,.7,1.4,3):
   w=np.exp(-G/(2*ell**2)); U=w.mean(2); yield ('add',flip,ell),U,E
  for ell in (.6,1.2,1.8,2.4,3):
   v=np.sin(f*ell); w=np.cos(f*ell); U=(v@v.T+w@w.T)/f.shape[1]; yield ('spectral',flip,ell),U,E
   for mode in (np.tanh,np.arctan):
    v=mode(f*ell); U=(v@v.T)/f.shape[1]; yield (mode.__name__,flip,ell),U,E

rows=[]; blend=[]
for name in ('z','Bz','Iz','0z','0Bz','1Bz','1z'):
 q=D[name]; z0=D['z']; L=1+z0@z0.T/4; E0=cdist(z0,z0,'sqeuclidean')
 for tag,U,E in bases(q):
  for p in (1,2,3):
   for h in (4,10,24):
    for noise in (1e-6,.0001,.002):
     k=(1+U)**p + np.exp(-E0/h)+L; es=[]; cv=[]; yh=[]
     for i,j in splits:
      ix=np.r_[i,j]; pred,r=_fit(k[np.ix_(ix,ix)],len(i),y[i],noise); es.append(np.mean(abs(pred-y[j]))); cv.append(np.mean(abs(r))); yh.append(pred)
     rows.append((np.round(es,5),np.round(cv,5),(name,tag,p,h,noise))); blend.append(yh)
for si in (0,1,2):
 key=[lambda r:r[0][0],lambda r:r[0][1:].mean(),lambda r:r[1][0]][si]; print('top',si,*sorted(rows,key=key)[:8],sep='\n')
for temp in (.001,.003,.006,.012):
 print('mix',temp,[round(np.mean(abs(np.average(np.array([v[i] for v in blend]),axis=0,weights=np.exp(-(np.array([r[1][i] for r in rows])-min(r[1][i] for r in rows))/temp))-y[j])),5) for i,(_,j) in enumerate(splits)])
