exec(open('/tmp/probe11.py').read().split('for k,p in')[0])
from scipy.optimize import least_squares
funcs={'erf':lambda z,w:erf(w*z),'tanh':lambda z,w:np.tanh(w*z),'sin':lambda z,w:np.sin(w*z),'rational':lambda z,w:z/(1+w*z*z),'softsign':lambda z,w:z/np.sqrt(1+w*z*z),'asinh':lambda z,w:np.arcsinh(w*z),'atan':lambda z,w:np.arctan(w*z)}
p=s['panel_a'];u=p['u'];v=p['v'];m=np.max(abs(v),axis=1)<3
for name,f in funcs.items():
 for deg in (1,2):
  a=polys(u,deg)
  def res(w):
   b=np.c_[a,f(u,w)][m];return np.sqrt(np.mean((b@fit(b,v[m])-v[m])**2))
  opt=minimize_scalar(res,bounds=(.1,8),method='bounded'); print('A',name,deg,round(opt.x,5),round(opt.fun,7),res(2.15),res(1.25))
p=s['panel_c'];u=p['u'];v=p['v1'];r=p['v2'];
for name,f in funcs.items():
 for deg in (1,2):
  a=polys(u,deg)
  def res(w):
   b=np.c_[a,f(u,w)];return np.sqrt(np.mean((b@fit(b,v)-v)**2))
  opt=minimize_scalar(res,bounds=(.1,8),method='bounded'); print('C1',name,deg,round(opt.x,5),round(opt.fun,7),res(2.15),res(1.25))
for k in range(2,7):
 a=np.c_[aff(u),*[u**j for j in range(2,k+1)]]; c=fit(a,r);print('C2',k,np.sqrt(np.mean((a@c-r)**2)));print(np.round(c,5))
