import sys,pickle,numpy as np
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_02
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fold(tr,va):
    t=s['train']; sub=dict(s,train={**t,**{k:t[k][tr] for k in ['x','y','traj']}},test={'x':t['x'][va]})
    p=node_00.run_node({'__input__':sub}); state=node_02.run_node({'node_00':p})['state_pred']; return p,state,t['y'][tr],t['y'][va],t['traj'][va],t['x'][tr],t['x'][va]
def expand(z,d):
    return np.column_stack([np.prod(z[:,i],axis=1) for i in combinations_with_replacement(range(z.shape[1]),d)])
if __name__=='__main__':
 for tr,va in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]:
    p,state,y,q,t,x,xq=fold(tr,va); m=p['modes']; a,b=p['scores'],state@m.T; sy,qy=p['sy'],p['qy']; st,qt=p['st'],p['qt']; print('trajectory mae',np.abs(t-state).mean(axis=0));
    for name,f,g in [('sy',sy,qy),('x',x,xq),('st',st,qt),('state',a,b),('sy+state',np.c_[sy,a],np.c_[qy,b]),('sy+st',np.c_[sy,st],np.c_[qy,qt]),('sy+state+st',np.c_[sy,a,st],np.c_[qy,b,qt]),('sy+state+state2',np.c_[sy,a,expand(a,2)],np.c_[qy,b,expand(b,2)]),('sy+state quad',np.c_[sy,a,expand(np.c_[sy,a],2)],np.c_[qy,b,expand(np.c_[qy,b],2)]),('sy degree 2',np.c_[sy,expand(sy,2)],np.c_[qy,expand(qy,2)]),('x degree 3',np.c_[x,expand(x,2),expand(x,3)],np.c_[xq,expand(xq,2),expand(xq,3)])]:
      f=np.c_[np.ones(len(f)),f]; g=np.c_[np.ones(len(g)),g]; w=np.linalg.lstsq(f,y,rcond=None)[0]; print(name,'train/cv/max',*[round(v,6) for v in (np.abs(y-f@w).mean(),np.abs(q-g@w).mean(),np.abs(q-g@w).max())], 'coef',np.round(w,3) if len(w)<15 else '')
