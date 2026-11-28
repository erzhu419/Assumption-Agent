import importlib.util,pickle,numpy as np
from scipy.optimize import minimize_scalar
import copy
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=s['eval_times']; T=tr['traj']; A=m._affine
p=s['panel_a']; d=A(p['u']); v=p['v']; coef=np.linalg.lstsq(d,v,rcond=None)[0]; print('calibration initial residual quantiles',np.quantile(np.linalg.norm(v-d@coef,axis=1),[0,.5,.8,.9,.95,1]));
for _ in range(30):
 e=np.linalg.norm(v-d@coef,axis=1); w=np.minimum(1,max(.001,1.2*np.median(e))/np.maximum(e,1e-12)); coef=np.linalg.lstsq(d*w[:,None],v*w[:,None],rcond=None)[0]
e=np.linalg.norm(v-d@coef,axis=1); keep=e<max(.004,4*np.median(e)); coef=np.linalg.lstsq(d[keep],v[keep],rcond=None)[0]; print('calibration',coef,'kept',keep.sum(),'resid',np.sqrt(np.mean((d[keep]@coef-v[keep])**2)))
sets=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(28).permutation(72),4))+list(np.array_split(np.random.default_rng(71).permutation(72),4)); sets.append(np.argsort(np.linalg.norm(x,axis=1))[-12:])
p=s['panel_b']; b=np.linalg.lstsq(A(p['u']),p['v'],rcond=None)[0]; p=s['panel_c']; cb=np.linalg.lstsq(A(p['u']+.45*p['u']**2),p['v2'],rcond=None)[0]; trans=lambda h:A(h+.45*h*h)@cb
z=np.linalg.solve(coef[1:].T,(x-coef[0]).T).T
base={'x':x,'calinv':z,'calfwd':A(x)@coef}
for k,g in list(base.items()):
 base[k+'b']=A(g)@b; base[k+'bi']=np.linalg.solve(b[1:].T,(g-b[0]).T).T
for k,g in list(base.items()):base[k+'c']=trans(g)
res={}; ar={}; te=[]; add=[]; mixes=[]; temporal=[]
for j in sets:
 i=np.setdiff1d(np.arange(72),j); order=np.r_[i,j]; n=len(i); a=A(x[order]); sf=copy.copy(s); sf['train']={k:(v[i] if k!='summary' else v) for k,v in tr.items()}; sf['test']={'x':x[j]}; pred=m.solve_batch(sf); te.append([np.mean(abs(pred['y_hat']-y[j])),np.mean(abs(pred['state_pred']-T[j])),np.mean(abs(pred['state_pred'][:,-1]-T[j,-1]))]);
 shape=lambda r:-np.expm1(-r*t); r=minimize_scalar(lambda r:np.sum((T[i]-np.outer(T[i]@shape(r)/(shape(r)@shape(r)),shape(r)))**2),bounds=(.05,4),method='bounded').x; ss=shape(r); amp=T@ss/(ss@ss); v0=np.linalg.svd(T[i]-np.outer(amp[i],ss),full_matrices=False)[2][0]; dr=amp*0; D=A(x); corr=D[j]@np.linalg.lstsq(D[i],(T@v0)[i],rcond=None)[0]; add.append(np.mean(abs(T[j]-pred['state_pred']-np.outer(corr,v0)),axis=0));
 yt={}
 for k,u in base.items():
  g=u[order]; q=m._terms(g,2);
  def reg(f,ridge,qw,targ):return m._fit([(a,1e4),(q,qw),(f,1.),(m._terms(f,2),1.)],n,targ[i],ridge)
  yp=np.mean([reg(np.sin(sc*g),rr,w,y) for sc,rr,w in ((.7,.02,1),(.85,.045,.55),(1.,.09,.7),(.7,.02,.15))],axis=0); yt[k]=yp; res.setdefault(k,[]).append(np.mean(abs(yp-y[j])));
  f=np.arctan(g); ap=m._fit([(a,1e4),(m._terms(x[order],2),.2),(m._terms(x[order],3),.02),(f,1.),(m._terms(f,2),.1)],n,amp[i],.015); ar.setdefault(k,[]).append(np.mean(abs(ap-amp[j])))
 rr=np.linalg.norm(x[j],axis=1); w=np.clip((rr-1.3)/.9,0,1); mixes.append([np.mean(abs((1-r)*yt["xb"]+r*yt["xbi"]-y[j])) for r in [0,.1,.2,.3,.5*w,.7*w]]); temporal.append([np.mean(abs(T[j]-pred["state_pred"]-f*np.outer(corr,v0)),axis=0) for f in [0,.4,.7,1]])
print('CURRENT y, state, terminal'); print(np.array(te)); print('ADD time residual avg, terminal',np.c_[np.mean(add,axis=1),np.array(add)[:,-1]]);
for k in base:
 e=np.array(res[k]); a=np.array(ar[k]); print(k,'y',e[0],e[1:-1].mean(),e[1:-1].max(),e[-1],'amplitude',a[0],a[1:-1].mean(),a[1:-1].max(),a[-1])
print('scalar mixes by split',np.array(mixes)); v=np.array(temporal); print('time grid designated / mean shuffled / worst shuffled / high norm'); print('all',np.array([v[0].mean(1),v[1:-1].mean((0,2)),v[1:-1].mean(2).max(0),v[-1].mean(1)])); print('last',np.array([v[0,:,-1],v[1:-1,:,-1].mean(0),v[1:-1,:,-1].max(0),v[-1,:,-1]]))
