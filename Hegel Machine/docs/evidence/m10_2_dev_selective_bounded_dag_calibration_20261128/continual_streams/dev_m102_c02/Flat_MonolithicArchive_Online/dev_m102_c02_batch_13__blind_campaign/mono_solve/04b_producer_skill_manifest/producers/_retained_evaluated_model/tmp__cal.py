import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.linalg.norm(v,axis=1)<5;u,v=u[keep],v[keep]; w=np.roll(u,-1,axis=1); uv=u*w
fs={'id':lambda a,p:a,'sin':lambda a,p:np.sin(p*a),'tanh':lambda a,p:np.tanh(p*a),'atan':lambda a,p:np.arctan(p*a),'cubic':lambda a,p:a**3,'rat':lambda a,p:a/(1+p*a*a),'square':lambda a,p:a*a}
def calib(a, v, params, names):
 w=np.roll(a,-1,axis=1); q=a*w
 return [aff(np.c_[fs[i](a[:,j],params[0]),fs[k](w[:,j],params[1]),fs[l](q[:,j],params[2])]) for j in range(4) for i,k,l in [names]]
def calc(p,names,out=False):
 h=calib(u,v,p,names); c=[fit(a,b) for a,b in zip(h,v.T)]; y=np.stack([a@b for a,b in zip(h,c)],1); return c if out else (y-v).ravel()
if __name__=='__main__':
 import itertools
 r=[]
 for ns in itertools.product(('id','sin','atan','tanh','rat'),repeat=3):
  p=least_squares(calc,[1.,1.,1.],args=(ns,),bounds=(.05,5),max_nfev=50); r.append((np.std(p.fun),ns,p.x,calc(p.x,ns,True)))
 for a in sorted(r,key=lambda q:q[0])[:10]:print(a)
 for key in ('v1','v2'):
  u=s['panel_c']['u'];v=s['panel_c'][key]; funcs={'soft':(lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.15,0],[.1,-3],[10,3]),'dead':(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.22,1.25],[0,.2],[1,3]),'rat':(lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.62,1.07,1.57],[0,.1,.1],[5,5,5]),'power':(lambda x,p:np.sign(x)*abs(x)**p[0],[1.25],[.1],[5])}
  for k,(fn,p,lo,hi) in funcs.items():
   def fun(p):
    a=aff(fn(u,p));return (a@fit(a,v)-v).ravel()
   r=least_squares(fun,p,bounds=(lo,hi)); print(key,k,r.x,np.std(r.fun))
