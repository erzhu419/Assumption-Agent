import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
from node_00 import run_node,_calibrate
from node_01 import run_node as targets
from node_02 import run_node as pred
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['x']); x=s['train']['x']; z=_calibrate(s['panel_a'],x); f=run_node({'__input__':s})['features'][:n]; y=targets({'__input__':s})['targets']; np.set_printoptions(precision=4,suppress=True)
print('z range',z.min(0),z.max(0),'target range',y.min(0),y.max(0),'f quant',np.quantile(f,[0,.1,.5,.9,1],axis=0),'row max',np.argsort(np.linalg.norm(f,axis=1))[-10:]); print('corr',np.corrcoef(np.c_[z,f,y].T)[-y.shape[1]:,:]);
def quad(a):return np.column_stack([a,*[a[:,i:i+1]*a[:,j:j+1] for i in range(a.shape[1]) for j in range(i+1)]])
def cub(a):return np.column_stack([quad(a),*[a[:,i]*a[:,j]*a[:,k] for i in range(a.shape[1]) for j in range(i+1) for k in range(j+1)]])
for k,a in {'x':x,'z':z,'f':f,'quad z':quad(z),'cubic z':cub(z),'quad f':quad(f),'z+f':np.c_[z,f],'quad z+f':quad(np.c_[z,f]),'quad x':quad(x)}.items():
 a=np.c_[np.ones(n),a]; sv=np.linalg.svd(a,compute_uv=False); b=np.linalg.lstsq(a[:54],y[:54],rcond=1e-8)[0]; e=a@b-y; print(k,'rank',sum(sv>sv[0]*1e-8),'train',np.mean(abs(e[:54]),0),'val',np.mean(abs(e[54:]),0),'corr ranges',np.max(abs(e[54:]),0));
