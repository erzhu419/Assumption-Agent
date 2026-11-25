import pickle,numpy as np
from scipy.optimize import least_squares
from prony_probe import basis,solve
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); x=p['experiment']; t=x['train_t']; e=x['train_strain']; y=x['train_stress_raw']; tt=x['test_t']; et=x['test_strain']; z=p['gt_test_stress']; d=np.column_stack((np.ones_like(t),(t-t.mean())/np.ptp(t))); A=basis(t,e,np.geomspace(.02,20,80),'ramp')
lo=np.r_[np.zeros(4),[-np.inf]*2,[np.log(.02)]*3]; hi=np.r_[[np.inf]*6,[np.log(20)]*3]
def fit(exclude,action,penalty):
 c,b,_,s=solve(A[~exclude],d[~exclude],y[~exclude]); r=y-A@c-d@b; r-=np.median(r[~exclude]); flags=abs(r)>6*s
 v0=np.r_[max(c[0],.001),25,15,10,b,np.log([.15,1.4,8])]; spike=flags&~exclude
 mag=np.median(abs(r[spike])); scatter=1.4826*np.median(abs(abs(r[spike])-mag)); clean=y.copy(); mask=~exclude&~flags
 if action=='correct' and sum(spike)>=6 and mag>30*s and scatter<2*s:
  clean[spike]-=np.sign(r[spike])*mag; mask=~exclude
 def train(v): return basis(t,e,np.exp(v[6:]),'ramp')@v[:4]+d@v[4:6]
 def fun(v): return np.r_[(train(v)-clean)[mask]/s,np.sqrt(penalty)*np.exp(v[6:])/(np.ptp(t)/3)]
 v=least_squares(fun,v0,bounds=(lo,hi),max_nfev=150,ftol=1e-9).x
 return v,train(v)-y,flags,s,mag,scatter
for action in ('mask','correct'):
 for penalty in (0,.5,1,2):
  v,r,flags,s,mag,sc=fit(np.zeros(t.size,bool),action,penalty); pred=basis(tt,et,np.exp(v[6:]),'ramp')@v[:4]; errs=[]; gs=[]
  for a,b in [(0,2),(2,6),(6,12),(12,18),(18,24),(24,31)]:
   j=(t>=a)&(t<b); w,res,_,ss,_,_=fit(j,action,penalty); j&=abs(res)<6*ss; errs.append(np.sqrt(np.mean(res[j]**2))); gs.append(w[0])
  print(action,penalty,'G',round(v[0],4),'test',round(np.linalg.norm(pred-z)/np.linalg.norm(z),6),'spike mag/scatter/count',np.round([mag,sc,sum(flags)],5),'blocked',np.round(errs,5),'blocked G',np.round(gs,3))
