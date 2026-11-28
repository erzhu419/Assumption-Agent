exec(open('/tmp/readout_probe.py').read().split('for name,p in')[0])
from scipy.optimize import least_squares
from itertools import product,combinations_with_replacement
lat=np.cbrt(q); raw=s['train']['x']; tar=np.c_[y,t@np.linalg.svd(t,full_matrices=False)[2][:2].T]; tar/=tar.std(0)
c=s['panel_c']; u=c['u']; print('panels',[(k,[(j,np.shape(a)) for j,a in v.items()]) for k,v in s.items() if isinstance(v,dict)]); print('latent std',lat.std(0),'raw std',raw.std(0),'c std',u.std(0),'range',u.min(0),u.max(0))
def profile(x,y,fn,init):
 def res(p):
  a=fn(x,p); return (a@np.linalg.lstsq(a,y,rcond=1e-10)[0]-y).ravel()
 o=least_squares(res,init,max_nfev=100); a=fn(x,o.x); return np.mean(o.fun**2)**.5,o.x,np.linalg.lstsq(a,y,rcond=1e-10)[0]
xs={'latent':lat,'cubic':q,'raw':raw,'B(latent)':aff(np.cos(f*lat))@co,'B(cubic)':aff(np.cos(f*q))@co}
for key in ['v1','v2']:
 for name,fn in {'atan':lambda x,p:aff(np.arctan(p[0]*x+p[1:])), 'tanh':lambda x,p:aff(np.tanh(p[0]*x+p[1:])), 'affatan':lambda x,p:np.c_[aff(x),np.arctan(p[0]*x+p[1:])], 'affcos':lambda x,p:np.c_[aff(x),np.cos(p[0]*x+p[1:])], 'polycos':lambda x,p:np.c_[aff(x),x**2,x**3,np.cos(p[0]*x+p[1:])], 'affsquash':lambda x,p:np.c_[aff(x),(p[0]*x+p[1:])/np.sqrt(1+(p[0]*x+p[1:])**2)]}.items():
  e,p,w=min([profile(u,c[key],fn,np.r_[g,np.zeros(4)]) for g in (1,1.3,2.15,2.6,3.6)],key=lambda a:a[0]); print(key,name,e,p,flush=True)
  if e<.001:
   for k,x in [('lat',lat),('q',q),('B',xs['B(latent)']),('BQ',xs['B(cubic)'])]: xs[f'{k}:{key}:{name}']=fn(x,p)@w

def feats(x,fn,p,mode):
 a=fn(p[:4]*x+p[4:8]); b=np.c_[aff(a),x] if 'x' in mode else aff(a)
 if '2' in mode: b=np.c_[b,*[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)]]
 return b
results=[]
for key,x in xs.items():
 for name,fn in [('cos',np.cos),('tanh',np.tanh),('atan',np.arctan)]:
  for mode in ['','x','2','x2']:
   for g in [1.3,2.15,2.6]:
    m=lambda x,p:feats(x,fn,p,mode); e,p,w=profile(x[:54],tar[:54],m,np.r_[np.full(4,g),np.zeros(4)]); valid=np.mean(abs(m(x[54:],p)@w-tar[54:]),0); results.append((valid.mean(),e,key,name,mode,p,valid))
for a in sorted(results,key=lambda a:a[0])[:20]:print(a)
