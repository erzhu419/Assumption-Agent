import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m99_probe import *
from scipy.optimize import least_squares
C=models['linear']; ycal=np.linalg.solve(C[1:].T,(raw.reshape(-1,4,3)-C[0][None,:,None]).transpose(1,0,2).reshape(4,-1)).reshape(4,-1,3).transpose(1,0,2).reshape(raw.shape)
print('affine sweep residual rms/max',np.sqrt(np.mean((feat(x,'linear')@C-y)**2)),np.max(np.abs(feat(x,'linear')@C-y)))
for k in 'xy':
 print(k,'kappa endpoints',s['kappa_'+k][[0,1,-2,-1]],'v endpoints',s['vel_'+k][[0,-1]],'h',s['L'+k]/(s['n'+k]-1))
def getaxis(ax,p,form):
 k=s['kappa_'+ax]; v=s['vel_'+ax]; h=s['L'+ax]/(len(k)-1); m=axis(k,v,h,0,'reflect').tolil()
 for a,(i,j,b) in enumerate([(0,1,len(k)-1),(len(k)-1,len(k)-2,0)]):
  edge=(k[i]+k[b])/2/h**2; inner=(k[i]+k[j])/2/h**2; adv=(-1 if a else 1)*v[i]/h
  if form=='fit': near,wrap,leak=p[3*a:3*a+3]
  elif form=='scale': near,wrap,leak=k[i]/h**2,edge*p[0],0
  else:
   near=(inner if 'face' in form else k[i]/h**2)+(max(-adv,0) if 'adv' in form else 0)
   wrap=(k[i]/h**2 if 'local' in form else edge)+(max(adv,0) if 'adv' in form else 0)
   leak=0
  m[i,j]=-near; m[i,b]=-wrap; m[i,i]=near+wrap+leak
 return m.tocsc()
def calc(form,p=()):
 n=len(p)//2
 return S_for(getaxis('x',p[:n],form),getaxis('y',p[n:],form))
def score(e): return np.sqrt(np.mean(e*e,axis=0)).round(8)
for f in ['base','face','local','adv','faceadv','localadv','facelocal','facelocaladv']:
 e=calc(f)-ycal; print(f,score(e),'projected',score(proj(e)))
for f,p in [('scale',np.ones(2)),('fit',np.concatenate([[s['kappa_'+a][i]/h**2,(s['kappa_'+a][0]+s['kappa_'+a][-1])/2/h**2,0] for a in 'xy' for h in [s['L'+a]/(s['n'+a]-1)] for i in [0,-1]]))]:
 for held in [None,0,1,2]:
  train=[i for i in range(3) if i!=held]; o=least_squares(lambda p:(calc(f,p)-ycal)[:,train].ravel(),p,gtol=1e-13,ftol=1e-12,xtol=1e-12); print(f,'held',held,'parameters',o.x.round(6),'RMSE',score(calc(f,o.x)-ycal),'jac singular min',np.linalg.svd(o.jac,compute_uv=False)[-1])
