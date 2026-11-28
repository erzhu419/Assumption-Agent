import importlib.util,pickle,numpy as np,shutil
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
p=importlib.util.spec_from_file_location('c','/tmp/candidate.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
T=np.c_[tr['y'],tr['traj']]
results=[]
for ix,te in [(np.arange(52),np.arange(52,72))]+[(p[:40],p[40:]) for p in [np.random.default_rng(i).permutation(52) for i in (13,31,43)]]:
 raw=np.r_[tr['x'][ix],tr['x'][te]]; x=m._calibrate(s['panel_a'],raw); c,d=m._branches(s['panel_c'],x); h=m._bias(np.c_[c,d]); p0=m._baseline(raw,h,T[ix]); p1=m._joint(s,raw,x,c,d,T[ix]);
 scores=[]
 for w in (0,.35,.7,1):
  e=np.abs((1-w)*p0+w*p1-T[te]);scores.append([e[:,0].mean(),e[:,1:].mean()]);
 results.append(scores); print('slice',len(ix),len(te),np.round(scores,6),flush=True)
a=np.array(results); w=(0,.35,.7,1)[np.argmin(np.max(a[1:],axis=0).sum(1))]; j=(0,.35,.7,1).index(w)
if np.all(a[0,-1]<.038): w=1
elif not np.all(a[0,j]<=a[0,0]): w=0
print('promotion weight',w)
if w:
 f=open('/tmp/candidate.py').read()
 if w!=1:f=f.replace('pred=_joint(study,raw,x,c,d,target)',f'pred={w}*_joint(study,raw,x,c,d,target)+{1-w}*_baseline(raw,h,target)')
 open('/holdout_task/batch_solver.py','w').write(f)
