import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_01,node_02
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); dev=np.arange(n-20); official=np.arange(n-20,n)
def build(train,valid):
 q=node_00.run_node({'__input__':{**s,'train':{k:t[k][train] for k in ('x','y','traj')},'test':{'x':t['x'][valid]}}}); p=node_01.run_node({'node_00':q})['poly']; b=node_02.run_node({'node_00':q}); return p,b['smooth'],b['vt'],t['y'][valid],t['traj'][valid]
def cross(ids,seed,k):
 return [build(np.setdiff1d(ids,v),v) for v in np.array_split(np.random.default_rng(seed).permutation(ids),k)]
if __name__=='__main__':
 outer=np.array_split(np.random.default_rng(962).permutation(dev),4); out=[]
 for j,v in enumerate(outer):
  a=np.setdiff1d(dev,v); out.append((sum([cross(a,r,5) for r in (44,109)],[]),build(a,v)))
 out.append((sum([cross(dev,r,6) for r in (44,109,203)],[]),build(dev,official)))
 pickle.dump(out,open('/tmp/node03_nested.pkl','wb'))
 for j,(_, (p,b,vt,y,tr)) in enumerate(out):
  z=(.4*p[:,1:]+.6*b[:,1:])@vt; dz=(p[:,1:]-b[:,1:])@vt
  print(j,'base',np.mean(abs(y-b[:,0])),np.mean(abs(z-tr)))
  if j==4: print(np.round(np.column_stack((y,b[:,0],y-b[:,0],p[:,0]-b[:,0],z[:,0],z[:,-1],dz[:,0],dz[:,-1])),4))
