import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a,b,c=[s['panel_'+j] for j in 'abc']
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(d,y):return np.linalg.lstsq(d,y,rcond=None)[0]
def profile(u,v,f,k):
 def design(k):return f(u,k)
 o=least_squares(lambda k:(design(k)@fit(design(k),v)-v).ravel(),k,bounds=(.005,8));return np.mean(o.fun**2)**.5,o.x,fit(design(o.x),v)
fun={'atan':lambda x,k:np.arctan(k*x),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'soft':lambda x,k:x/np.sqrt(1+k*x*x),'rat':lambda x,k:x/(1+k*x*x),'asinh':lambda x,k:np.arcsinh(k*x)}
u=a['u'];v=a['v'];m=np.linalg.norm(v-u,axis=1)<4;u,v=u[m],v[m]; auv=u,v
for lab,(u,v) in [('A',auv),('C',(c['u'],c['v1']))]:
 for key,f in fun.items():
  for add in (False,True):
   des=lambda x,k:np.c_[aff(x),f(x,k)] if add else aff(f(x,k)); r,k,w=profile(u,v,des,[1.5]*4);print(lab,key,add,round(r,7),k.round(4));
   if r<.001:print(w.round(4))
