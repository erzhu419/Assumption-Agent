import pickle,sys
sys.path.insert(0,'/holdout_task')
import batch_solver as b
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,t=d['x'],d['y'],d['traj']; n=len(x)
z=b._calibrate(s['panel_a'],x); g,h=np.split(b._panel_maps(s['panel_c'],z),2,1)
print('cal ranges',z.min(0),z.max(0)); c=s['panel_c']; print('panel errors',np.std(b._panel_maps(c,c['u'])-np.c_[c['v1'],c['v2']],axis=0)); print('summary vs moments',d['summary'],[y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()]);
rng=np.random.default_rng(123); perms=[np.arange(n)]+[rng.permutation(n) for _ in range(6)]; preds=[]
for p in perms:
 a,q=p[:52],p[52:]; out=b.solve_batch(dict(s,train=dict(x=x[a],y=y[a],traj=t[a]),test=dict(x=x[q]))); preds.append([abs(y[q]-out['y_hat']).mean(),abs(t[q]-out['state_pred']).mean()]);
print('baseline slices',np.array(preds))
F=b._time_basis(s['eval_times'],t[:52]); D=np.c_[y,b.lin(F,t.T).T]; r=[]
def check(name,f,regs=(.001,.05,.2)):
 for reg in regs:
  vals=[]
  for p in perms:
   a,q=p[:52],p[52:]; pred=b._readout(f[p],D[a],52,reg); err=pred-D[q]; vals.append([abs(err[:,0]).mean(),abs(err[:,1:]@F.T).mean()]);
  vals=np.array(vals); r.append((name,reg,vals[0],vals.mean(0),vals.max(0)))
base=np.c_[g,h]; prod=g*h; dot=prod.sum(1,keepdims=True); bmap=b.lin(b.one(s['panel_b']['u']),s['panel_b']['v']); w=b.one(z)@bmap
for name,f in [('base',base),('dot',np.c_[base,dot]),('diag',np.c_[base,prod]),('zdiag',np.c_[base,z,prod]),('wdiag',np.c_[base,w,prod]),('cross',np.c_[base,(g[:,:,None]*h[:,None,:]).reshape(n,16)])]:check(name,f)
for q in [.22,.44,1.25]:
 for den,k in [('g',g),('h',h),('g+h',g+h),('g-h',g-h),('dot',dot)]:
  for nm,u in [('g',g),('h',h),('dot',dot),('prod',prod)]:
   transformed=u/(1+q*k*k); check((q,den,nm),np.c_[base,prod,transformed]);check((q,den,nm,'compact'),np.c_[base,transformed]);
 for nm,u in [('g',g),('h',h),('g+h',g+h),('g-h',g-h),('prod',prod),('dot',dot),('z',z)]:
  check((q,nm,'tanh'),np.c_[base,prod,np.tanh(q*u)]);check((q,nm,'soft'),np.c_[base,prod,u/(1+q*abs(u))]);
for i in [0,1]:print('best',i,*sorted(r,key=lambda a:a[3][i]+.35*a[4][i])[:10],sep='\n')
