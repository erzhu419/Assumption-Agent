import pickle,sys,importlib.util,numpy as np
sys.path.insert(0,'/tmp')
import node_next as m
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];y=s['train']['y'];T=s['train']['traj'];n=52; a=np.arange(n); b=np.arange(n,len(y)); rng=np.random.default_rng(225);folds=[(np.setdiff1d(a,j),j) for j in np.array_split(rng.permutation(n),4)];
cs=m._coords(s,f.run_node({'__input__':s}));Y=np.c_[y,T];
for i,x in enumerate(cs):
 print('COORD',i,flush=True);Q=m._quad(x[:len(y)]);e=Y-Q@m._ls(Q,Y); print('residual correlations',np.corrcoef(e.T)[0].round(3),'y residual extreme',sorted(abs(e[:,0]))[-10:],flush=True)
 for mode,key in [(0,(0,0.)),(1,(1,.05)),(2,(2,.06)),(3,(3,0)),(4,(4,0)),(5,(5,0)),(6,(6,0)),(7,(7,0)),(8,(8,0)),(9,(9,0)),(17,(17,.01))]:
  es=[]
  for tr,te in folds+[(a,b)]:
   scal=m._fit_model(x[tr],y[tr],x[te],key); ts=m._fit_model(x[tr],T[tr],x[te],key)
   es.append([abs(scal-y[te]).mean(),abs(ts-T[te]).mean()])
  print(i,mode,np.mean(es[:4],axis=0).round(5),np.array(es[-1]).round(5),flush=True)
