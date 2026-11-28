from fit_dag05 import *
from scipy.optimize import least_squares
from functools import partial
FUN={'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+(k*x)**2),'atan':lambda x,k:np.arctan(k*x),'soft':lambda x,k:x/np.sqrt(1+(k*x)**2),'cos':lambda x,k:np.cos(k*x),'gauss':lambda x,k:np.exp(-(k*x)**2),'erat':lambda x,k:1/(1+(k*x)**2)}
def F(x,f,k,linear=True):
 return np.column_stack((np.ones(len(x)),*((x,) if linear else ()),FUN[f](x,k)))
def fit(x,Y,f,linear=True):
 def res(k):
  a=F(x,f,k,linear);return (Y-a@np.linalg.lstsq(a,Y,rcond=None)[0]).ravel()
 best=min([least_squares(res,np.repeat(k,4),bounds=(.06,9),max_nfev=150) for k in (.6,1.5,2.8,4.5,6.2)],key=lambda o:o.cost)
 return best.x,np.sqrt(2*best.cost/Y.size)
if __name__=='__main__':
 pc=raw['panel_c']; u=pc['u'];
 for key,x,Y,fs in [('v1',u,pc['v1'],list(FUN)[:5]),('v2',u,pc['v2'],list(FUN)[5:]),('y',z,y,list(FUN)[:5]),('T',z,T,list(FUN)[:5])]:
  print(key,flush=True)
  for f in fs:
   for lin in (False,True):
    if not lin and key=='v2':continue
    k,r=fit(x,Y,f,lin);print(f,lin,np.round(k,5),r,score(F(x,f,k,lin),Y),flush=True)
