from struct13 import *
from scipy.optimize import least_squares
for name in ('panel_a','panel_b','panel_c'):
 print(name,[(k,v.shape,np.min(v),np.max(v)) for k,v in s[name].items()]); print(str(s[name])[:300])
print('times',s['eval_times'],'keys',s.keys())
print('ranges z',z.min(0),z.max(0)); print('first x,z,y,T',np.c_[x,z,y,T[:,-1]][:8]); print('panel train match',[np.min(np.linalg.norm(z[:,None]-s[k]['u'][None],axis=2)) for k in ('panel_a','panel_b','panel_c')])
for name in ('panel_b','panel_c'):
 u=s[name]['u']; v=np.column_stack([a for k,a in s[name].items() if k!='u']); print(name,'N',len(u));
 def branch(a,h,kind):
  if kind==0:return aff(np.sin(h[0]*a))
  if kind==1:return np.c_[aff(a),np.sin(h[0]*a),np.cos(h[0]*a)]
  if kind==2:return np.c_[aff(a),np.tanh(h[0]*a),np.sin(h[1]*a)]
  if kind==3:return np.c_[aff(a),np.sin(h[0]*a),np.cos(h[0]*a),np.sin(h[1]*np.roll(a,-1,axis=1)*a)]
  if kind==4:return np.c_[aff(a),np.sin(h[0]*a),np.cos(h[0]*a),a*np.roll(a,-1,axis=1)]
  if kind==5:return np.c_[aff(a),np.sin(h[0]*a),np.cos(h[1]*a)]
 for kind in range(6):
  def res(h):
   A=branch(u,h,kind); return (A@fit(A,v)-v).ravel()
  os=[least_squares(res,[i,j],bounds=(.08,12),max_nfev=200) for i,j in it.product((.5,1.5,2.5,4,6),repeat=2)]; o=min(os,key=lambda o:o.cost); A=branch(u,o.x,kind); b=branch(z,o.x,kind)@fit(A,v); print('warp',kind,o.x,np.mean(abs(o.fun)),scores(aff(b),y,.001),scores(aff(b),T,.001),scores(poly(b,2),y,.01),scores(poly(b,2),T,.01))
 for d in (2,3):
  A=poly(u,d); b=poly(z,d)@fit(A,v); print('pol warp',d,scores(aff(b),y,.001),scores(aff(b),T,.001),scores(poly(b,2),y,.01),scores(poly(b,2),T,.01))
