import sys, pickle, numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; z,c,d=m._coordinates(s,t['x']); p=s['panel_c']; u=p['u']; af=m._aff(u)
ops=[]
for key,fun,start in [('v1',np.tanh,2.3),('v2',np.cos,1.27)]:
 y=p[key]; options=[]
 for spec in ['shared','freq','shift','freqshift']:
  n=1 if spec=='shared' else 4; mshift=0 if spec in ['shared','freq'] else 4
  def des(x,a):return np.c_[m._aff(x),fun(a[:n]*x+(a[n:] if mshift else 0))]
  def err(a,idx):
   D=des(u[idx],a); return (y[idx]-D@m._fit(D,y[idx])).ravel()
  ini=np.r_[np.full(n,start),np.zeros(mshift)]; bounds=(np.r_[np.full(n,.6),np.full(mshift,-.6)],np.r_[np.full(n,4.),np.full(mshift,.6)])
  e=[]
  for i in range(len(u)):
   idx=np.arange(len(u))!=i; a=least_squares(err,ini,args=(idx,),bounds=bounds,max_nfev=100).x
   e.extend(y[i]-des(u[i:i+1],a)@m._fit(des(u[idx],a),y[idx]))
  idx=np.arange(len(u)); a=least_squares(err,ini,args=(idx,),bounds=bounds,max_nfev=100).x
  rms=np.sqrt(np.mean(np.square(e))); options.append((rms,des(z,a)@m._fit(des(u,a),y))); print('operator',key,spec,'LOO',round(rms,6),'params',a.round(4))
 ops.append(min(options,key=lambda a:a[0])[1])
sp=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=123).split(z),*KFold(4,shuffle=True,random_state=27).split(z)]
for name,cols in [('shared',(z,c,d)),('panel-LOO',(z,*ops))]:
 q=np.c_[tuple(cols)]; f=m._quadratic(q); base=np.c_[*[m._quadratic(a) for a in cols],f[:,12:]]
 for kind,F in [('clean',f),('weighted',base)]:
  for a in [.00015,.00045,.0009]:
   e=[]
   for i,j in sp:
    h,amp=m._modes(s['eval_times'],t['traj'][i]); Y=np.c_[t['y'][i],amp]; out=m._readout(F[np.r_[i,j]],Y,len(i),a); e.append([np.mean(abs(out[:,0]-t['y'][j])),np.mean(abs(out[:,1:]@h.T-t['traj'][j]))])
   e=np.array(e); print(name,kind,a,'fixed',e[:2].round(4).tolist(),'shuffled mean/max',e[2:].mean(0).round(4),e[2:].max(0).round(4))
