import pickle,numpy as np,sys
from itertools import combinations
sys.path.insert(0,'/holdout_task')
import node_00 as m
from _frozen_nodes import node_03 as sink, node_01 as basis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u'];v=p['v1']; n=len(u); C=lambda x:m._operator(p,'v2',x,False); B=lambda x:m._operator(s['panel_b'],'v',x,True)
x=m._calibrate(s['panel_a'],s['train']['x']); raw=s['train']['x']; b=B(x)
print('telemetry',s['telemetry_vec'],'range',x.min(),x.max()); print('C u,v1,v2 first',np.round(np.c_[u,v,p['v2']][:5],3)); print('B/C inputs',np.round(s['panel_b']['u'][:3],3), 'u diff',np.max(abs(s['panel_b']['u']-u)))
print('C v1 correlations',np.round(np.corrcoef(np.c_[u,v].T)[:4,4:],2))
for tag,z in [('u',u),('B(u)',B(u)),('v2',p['v2']),('cal(u)',m._calibrate(s['panel_a'],u))]:
 for name,zz in [('linear',z),('all2',np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]),('sin',np.sin(2.4*z)),('cos',np.cos(2.6*z)),('tanh',np.tanh(1.25*z)),('sign',np.sign(z)),('interact',np.c_[u,z,(u[:,:,None]*z[:,None,:]).reshape(n,-1)])]:
  D=np.c_[np.ones(n),zz]; q=np.linalg.svd(D,full_matrices=False); q=q[0][:,:np.linalg.matrix_rank(D)]; r=v-q@(q.T@v); h=1-(q*q).sum(1); print(tag,name,round(np.sqrt(np.mean(r*r)),4),round(np.sqrt(np.mean((r/np.maximum(h,1e-8)[:,None])**2)),4))
y=s['train']['y']; t=s['train']['traj']; splits=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]; coords={'x':x,'raw':raw,'b':b,'c(x)':C(x),'c(b)':C(b),'b(c(x))':B(C(x)),'c(raw)':C(raw),'b(raw)':B(raw),'c(b(raw))':C(B(raw))};
def score(xx,cc):
 D=basis._design(xx,cc); a,z=sink._features(xx,cc,D); out=[]
 for tr,va in splits:
  yp=sink._scalar(a[tr],z[tr],y[tr],a[va],z[va]); modes=np.linalg.svd(t[tr],full_matrices=False)[2][:2]; tp=D[va]@np.linalg.lstsq(D[tr],t[tr]@modes.T,rcond=None)[0]@modes; out.append((np.mean(abs(yp-y[va])),np.mean(abs(tp-t[va]))))
 return np.array(out)
r=[]
for xn,xx in coords.items():
 for cn,cc in coords.items():
  val=score(xx,cc);r.append((val.mean(),xn,cn,np.round(val,4).tolist()))
print('best coordinate pairings',*sorted(r)[:15],sep='\n')
print('best trajectory',*sorted(r,key=lambda row:np.array(row[3])[:,1].mean())[:5],sep='\n')
