import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for name in ['panel_a','panel_b','panel_c','train','test']:
 print(name,[(k,v.shape,(v.min(),v.max())) for k,v in s[name].items()]);
for name,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[name]['u'],s[name][k]; one=np.ones((len(u),1)); i,j=np.triu_indices(4)
 def ff(a): return np.linalg.lstsq(a,v,rcond=None)[0]
 for a in [np.c_[one,u,u*u],np.c_[one,u,u*u,u**3],np.c_[one,u,u[:,i]*u[:,j],u**3]]:
  print(name,k,a.shape[1],np.sqrt(np.mean((a@ff(a)-v)**2)))
 for kind in ('sin','cos','tanh','atan','rat','sinq'):
  def warp(p):
   f,g,h=p
   w={'sin':lambda:np.sin(f*u),'cos':lambda:np.cos(f*u),'tanh':lambda:np.tanh(f*u),'atan':lambda:np.arctan(f*u),'rat':lambda:u/(1+f*u*u),'sinq':lambda:np.sin(f*u+g)}[kind]()
   return np.c_[one,u+h*w] if kind!='sinq' else np.c_[one,u+h*w]
  def res(p):
   a=warp(p);return (a@ff(a)-v).ravel()
  o=min([least_squares(res,[f,g,h],bounds=([.05,-3,-8],[12,3,8]),max_nfev=85) for f in (.65,2,3.8,6) for g in ((0,.8,2) if kind=='sinq' else (0,)) for h in (-.4,.4)],key=lambda q:q.cost)
  print(kind,o.x.round(4),np.sqrt(np.mean(o.fun**2)).round(6))
