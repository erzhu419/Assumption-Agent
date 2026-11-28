exec(open('/tmp/probe04.py').read().split('A,err,keep=')[0])
from scipy.linalg import svd
A,*_=fitpanel(s['panel_a'],'v'); B,*_=fitpanel(s['panel_b'],'v')
P=s['panel_c']; u=P['u']; U=aff(u); V=np.c_[P['v1'],P['v2']]; funcs={'sq':lambda x:x*x,'cube':lambda x:x**3,'sin':np.sin,'tanh':np.tanh,'sat':lambda x:x/(1+x*x),'cos':np.cos}
for k,g in funcs.items():
 for j in [0,1]:
  H=np.c_[U,g(u)] if j==0 else np.c_[U,g(U@np.linalg.lstsq(U,V,rcond=None)[0])]; W=np.linalg.lstsq(H,V,rcond=None)[0]; e=H@W-V; print('C',k,j,np.sqrt(np.mean(e**2,axis=0)).round(5));
  if j==0 and k in ('sq','cube'):print(W[5:].round(3))
for k in [2,3]:
 from itertools import combinations_with_replacement as cr
 H=np.column_stack([np.prod(u[:,i],axis=1) for j in range(k+1) for i in cr(range(4),j)]); w=np.linalg.lstsq(H,V,rcond=None)[0]; print('polyC',k,np.sqrt(np.mean((H@w-V)**2,axis=0)).round(5));
 if k==2:print('quad C',w[5:].round(3))
for j in range(8):
 v=V[:,j]; w=np.linalg.lstsq(U,v,rcond=None)[0]; print('Ccoord',j,end=' ')
 for name,g in {'tanh':np.tanh,'sin':np.sin,'rat':lambda p:p/(1+.2*p*p),'quad':lambda p:p*p,'cube':lambda p:p**3}.items():
  def resid(p):
   H=np.c_[U,g(U@p)]; return H@np.linalg.lstsq(H,v,rcond=None)[0]-v
  sol=min([least_squares(resid,w*c,max_nfev=100) for c in [.5,1,2]],key=lambda p:p.cost); print(name,round(np.sqrt(2*sol.cost/len(u)),6),np.round(sol.x,2),end=' ')
 print()
def exp(t,r):return -np.expm1(-t*r)
for name,fun,p0 in [('exp',lambda p:exp(t,p[0]),[.8]),('expdiff',lambda p:exp(t,p[0])+p[1]*t,[.8,0]),('exp2',lambda p:exp(t,p[0])+p[1]*exp(t,p[2]),[.8,-.1,2]),('shift',lambda p:exp(t+p[1],p[0]),[.8,0]),('power',lambda p:exp(t,p[0])**p[1],[.8,1])]:
 def res(p):
  f=fun(p); return z-np.outer(z@f/(f@f),f)
 sol=least_squares(lambda p:res(p).ravel(),p0,max_nfev=400); print('time',name,sol.x,np.sqrt(np.mean(res(sol.x)**2,axis=0)).round(5),np.linalg.svd(res(sol.x),compute_uv=False).round(4))
f=np.linalg.svd(z,full_matrices=False)[2][0]; a=z@f; print('residual scores corr X, A, y',np.corrcoef(np.c_[np.linalg.svd(z,full_matrices=False)[0][:,:4],x,y].T)[:4,4:]);
