import sys,pickle,importlib
import numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
a,b,c=[importlib.import_module('node_%02d'%i).run_node for i in range(3)]
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=st['train']; n=len(t['y']); p=n-20

def predict(tr,te):
 sh=a({'__input__':dict(st,train={q:t[q][tr] for q in ('x','y','traj')},test={'x':t['x'][te]})}); po=b({'node_00':sh})['poly']; so=c({'node_00':sh}); sm=so['smooth']; return po[:,0],sm[:,0],(.4*po[:,1:]+.6*sm[:,1:])@so['vt']
rng=np.random.default_rng(556)
D=[]
for r in range(5):
 for ix in np.array_split(rng.permutation(p),5):
  P,S,stp=predict(np.setdiff1d(np.arange(p),ix),ix); D.extend(zip(ix,P,S,t['y'][ix],*stp.T))
A=np.array(D); P,S,stp=predict(np.arange(p),np.arange(p,n)); V=np.column_stack((np.arange(p,n),P,S,t['y'][p:],stp)); np.savez('/tmp/cal_sink.npz',A=A,V=V)
