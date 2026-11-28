import sys,itertools,pickle,numpy as np
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
from node_00 import _a, _b
from node_01 import run_node as read
from b8 import fit
from scipy.optimize import least_squares
from scipy.special import erf, expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; x,y,t=[s['train'][k] for k in ('x','y','traj')]; a=s['panel_a']; z=_a(a['u'],a['v'],x)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
forms={
 'rats':(lambda z,p:z**3/(1+p[0]*z*z)+p[1]*z/(1+p[2]*z*z),[.8,0,12],[0,-2,1],[10,2,100]),
 'odd_rat':(lambda z,p:(z**3+p[0]*z)/(1+p[1]*z*z+p[2]*z**4),[0,.8,0.01],[-.5,0,0],[.5,10,10]),
 'odd_rat5':(lambda z,p:(z**3+p[0]*z+p[1]*z**5)/(1+p[2]*z*z+p[3]*z**4),[0,0,.8,.01],[-.5,-.5,0,0],[.5,2,15,15]),
 'gauss':(lambda z,p:z*(1+p[0]*np.exp(-p[1]*z*z))+p[2]*erf(p[3]*z),[-.2,12,-.7,1.66],[-2,.1,-3,.2],[2,100,3,5]),
 'arcsinh':(lambda z,p:np.arcsinh(p[0]*z)+p[1]*z+p[2]*np.arcsinh(p[3]*z),[1.5,-1.5,-.1,7],[.3,-8,-4,2],[4,5,4,40]),
 'softcube':(lambda z,p:z*(1-np.exp(-p[0]*z*z))+p[1]*z+p[2]*z*np.exp(-p[3]*z*z),[1.,0,.1,15],[.1,-1,-1,1],[6,1,1,100]),
 'ratroot':(lambda z,p:z-z/np.sqrt(1+p[0]*z*z)+p[1]*z+p[2]*z/(1+p[3]*z*z),[1.,0,.1,15],[.02,-1,-1,1],[10,1,1,100]),
 'softabs':(lambda z,p:z*(np.sqrt(1+p[0]*z*z)-1)+p[1]*z+p[2]*z/(1+p[3]*z*z),[1.,0,.1,15],[.02,-1,-1,1],[30,1,1,100])}
def fitf(name,a,b):
 if name not in forms:return fit(name,a,b)[0]
 f,p0,lo,hi=forms[name]; basis=lambda x,p:np.c_[np.ones(len(x)),f(x,p)]
 def r(p):
  d=basis(a,p);return (d@ls(d,b)-b).ravel()
 p=least_squares(r,p0,bounds=(lo,hi),max_nfev=400).x
 return lambda x:basis(x,p)@ls(basis(a,p),b)
def score(w,k):
 idx=np.r_[np.where(k)[0],np.where(~k)[0]]; q=read({'__input__':{'train':{'y':y[k],'traj':t[k]}},'node_00':{'w':w[idx]}});return [np.mean(abs(q['y_hat']-y[~k])),np.mean(abs(q['state_pred']-t[~k]))]
for name in ['baseline','ecomr','scom5','acomr','ecom','scom','eplain','rplain',*forms]:
 es=[]
 for seed in range(4):
  k=np.arange(len(u))%4!=seed; pred=(lambda x:_b(u[k],v[k],x)) if name=='baseline' else fitf(name,u[k],v[k]); es.append(np.mean(abs(pred(u[~k])-v[~k])))
 pred=(lambda x:_b(u,v,x)) if name=='baseline' else fitf(name,u,v); w=pred(z)
 cv=np.array([score(w,np.arange(len(x))<i) for i in (48,50,52,54,56,58,60)]); kfold=np.array([score(w,np.arange(len(x))%4!=i) for i in range(4)])
 print(name,'B avg/max/fit',np.round([np.mean(es),max(es),np.mean(abs(pred(u)-v))],6),'y-prefix',cv[:,0].round(6),'prefix worst',cv.max(0).round(6),'interleaved worst',kfold.max(0).round(6),flush=True)
