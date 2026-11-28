import os,pickle,runpy,sys,copy,numpy as np
sys.path.insert(0,'/tmp');from research14 import _candidate
os.chdir('/holdout_task');s=pickle.load(open('public_batch.pkl','rb'))[0]['study'];run=runpy.run_path('_frozen_nodes/node_00.py')['run_node'];scores={}
for ids in (np.arange(52,72),np.arange(0,20),np.arange(20,40)):
 tr=np.setdiff1d(np.arange(72),ids);st=copy.copy(s);st['train']={k:s['train'][k][tr] for k in ('x','y','traj')};st['test']={'x':s['train']['x'][ids]};Y=np.c_[s['train']['y'][ids],s['train']['traj'][ids]]; models=_candidate(st,run({'__input__':st}),True); arr=[]
 for j,(q,tag,e,p) in enumerate(models):
  E=np.abs(Y-p); m=np.array([E[:,0].mean(),E[:,1:].mean()]); scores.setdefault(tag,[]).append(m);arr.append((m.sum(),tag,m))
 print('outer',[(tag,np.round(m,5)) for _,tag,m in sorted(arr,key=lambda x:x[0])[:10]], 'top internal', [(tag,np.round(m,5)) for _,tag,m in arr[:8]],flush=True)
ranked=sorted([(np.max(np.array(m)/[.038,.038]),tag,np.round(m,5)) for tag,m in scores.items()],key=lambda a:a[0]);print('worst',ranked[:12]);
