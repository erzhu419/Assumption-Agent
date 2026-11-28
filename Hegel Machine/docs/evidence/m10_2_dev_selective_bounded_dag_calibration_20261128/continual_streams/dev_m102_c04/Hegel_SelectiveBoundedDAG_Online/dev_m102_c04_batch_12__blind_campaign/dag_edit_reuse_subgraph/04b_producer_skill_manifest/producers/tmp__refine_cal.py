from profile_spatial import *
from scipy.optimize import least_squares
u,v=[s['panel_a'][k] for k in ('u','v')]; m=np.linalg.norm(u-v,axis=1)<4.; u,v=u[m],v[m]; shift=[2,3,0,1]; forms=dict(sin=lambda x,k:np.sin(x*k),tanh=lambda x,k:np.tanh(x*k),rat=lambda x,k:x/(1+k*x*x))
for mode in ['shared','two','eight','shared_plus','shift_sin']:
 for nm,g in forms.items():
  if mode=='shift_sin' and nm!='sin':continue
  def design(p,j):
   a,b=u[:,j],u[:,shift[j]]
   if mode=='shared':k,l=p[0],p[0]
   elif mode=='eight':k,l=p[j],p[4+j]
   else:k,l=p[:2]
   extra=[g(a,k),g(b,l)]
   if mode=='shared_plus':extra.extend([np.sin(p[2]*(a+b)),np.sin(p[2]*(a-b))])
   if mode=='shift_sin':extra=[np.sin(k*a),np.sin(l*b+p[2]*a)]
   return np.column_stack([np.ones(len(a)),a,b,*extra])
  def r(p):
   return np.column_stack([design(p,j)@ls(design(p,j),v[:,j])-v[:,j] for j in range(4)])
  best=min((least_squares(lambda p:r(p).ravel(),np.full({'shared':1,'two':2,'eight':8,'shared_plus':3,'shift_sin':3}[mode],x),max_nfev=400) for x in [.35,1.25,2.15,2.5,3.]),key=lambda z:z.cost)
  print(mode,nm,np.round(best.x,5), np.sqrt(np.mean(best.fun**2)), 'param',np.round(ls(design(best.x,0),v[:,0]),5))
