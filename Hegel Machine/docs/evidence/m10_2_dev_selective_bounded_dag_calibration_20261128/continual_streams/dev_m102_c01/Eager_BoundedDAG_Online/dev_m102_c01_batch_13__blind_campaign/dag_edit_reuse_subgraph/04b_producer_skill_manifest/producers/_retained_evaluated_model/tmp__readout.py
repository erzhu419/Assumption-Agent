import pickle,sys,numpy as np
from scipy.linalg import solve
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as a,node_02 as b
p=pickle.load(open('/holdout_task/public_batch.pkl','rb')); s=p[0]; print('structure',[(k,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in s.items()]); s=s.get('study',s)
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(y)
print('scales',x.min(0),x.max(0),y.min(),y.max(),np.linalg.svd(t,compute_uv=False)); print('cal',a._calibration(s['panel_a']));
D=dict(s); D['test']={'x':x}; p=a.run_node({'__input__':D}); u=p['sy']; v=p['st'][:,:4]; z=p['scores'];
def poly(x,d):
 return np.column_stack([x[:,i].prod(1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def ev(name,f):
 f=np.c_[np.ones(n),f]; w=np.linalg.lstsq(f,y,rcond=1e-9)[0]; r=y-f@w; oo=[]
 for k in [3,0,1,2]:
  j=np.arange(52,72) if k==3 else np.array_split(np.arange(n),3)[k]; i=np.setdiff1d(np.arange(n),j); wj=np.linalg.lstsq(f[i],y[i],rcond=1e-9)[0]; oo.append(np.mean(abs(y[j]-f[j]@wj)))
 return (round(np.mean(abs(r)),5),np.round(oo,5),np.round(w,3) if f.shape[1]<12 else '')
for name,f in dict(x=x,sy=u,st=v,score=z,xsy=np.c_[x,u],syst=np.c_[u,v],syscore=np.c_[u,z],xt=np.c_[x,z],syt=np.c_[u,t]).items():
 for d in ([1,2,3] if f.shape[1]<=4 else [1,2]): print(name,d,ev(name,poly(f,d)))
print('sy rows',u[:5]); print('y rows',y[:10]); print('score rows',z[:5]);
