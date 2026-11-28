import pickle,runpy,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); Y=np.c_[t['y'],t['traj']]; m=runpy.run_path('/holdout_task/node_00.py'); x,b,c,d=[z[:n] for z in m['_features'](s)]; ids=np.arange(n); vals=[ids[-20:]]+list(np.array_split(np.random.default_rng(63).permutation(n),4))+[np.argsort(np.linalg.norm(x,axis=1))[-15:]]
f=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
np.set_printoptions(precision=4,suppress=True,linewidth=135)
def err(p,v):
 e=abs(p-Y[v]);return [e[:,0].mean(),e[:,1:].mean()]
def report(es):
 e=np.array(es);return np.round(np.r_[e[0],e[1:5].max(0),e[-1]],4)
def score(h,mode='linear'):
 es=[]
 for v in vals:
  tr=np.setdiff1d(ids,v)
  if mode=='linear':
   a=np.c_[np.ones(n),h]; p=a[v]@np.linalg.lstsq(a[tr],Y[tr],rcond=1e-6)[0]
  else:
   o=f({'node_00':{'h':h[np.r_[tr,v]],'n_train':len(tr),'y':t['y'][tr],'traj':t['traj'][tr]}}); p=np.c_[o['y_hat'],o['state_pred']]
  es.append(err(p,v))
 return report(es)
p=x*np.roll(x,-1,axis=1); q=x*x; z=np.cos(2.4*x); core=np.c_[x,p,z]; base=np.c_[b,c,d]; mix=[b*c,b*d,c*d]; pair=[(b[:,:,None]*c[:,None,:]).reshape(n,-1),(b[:,:,None]*d[:,None,:]).reshape(n,-1),(c[:,:,None]*d[:,None,:]).reshape(n,-1)]; mats={'base':base,'diags':np.c_[base,*mix],'bc cd diag':np.c_[base,mix[0],mix[2]],'bc only all':np.c_[base,pair[0]],'cd only all':np.c_[base,pair[2]],'bc all cd diag':np.c_[base,pair[0],mix[2]],'bc diag cd all':np.c_[base,mix[0],pair[2]],'all interactions':np.c_[base,*pair],'primitive':core,'primitive plus squares':np.c_[core,q],'saturated bcd':np.c_[np.tanh(b),np.tanh(c),np.tanh(d)],'base and tanh':np.c_[base,np.tanh(base)],'gated':np.c_[base,c*np.tanh(d),b*np.tanh(c),b*np.tanh(d)],'base+triples':np.c_[base,*mix,b*c*d]}
for k,h in mats.items():
 print(k,h.shape[1],score(h),score(h,'sink') if h.shape[1]<27 else '')
for name,h in [('phys_pair',np.c_[base,*mix]),('primitive',core),('raw+cal',np.c_[t['x'],core])]:
 F=np.c_[h,np.column_stack([h[:,i]*h[:,j] for i,j in combinations_with_replacement(range(h.shape[1]),2)])]; D=(F-F.mean(0))/np.maximum(F.std(0),1e-12)
 for k in [6,10,15,22,32,40]:
  es=[]; picked=[]
  for v in vals:
   tr=np.setdiff1d(ids,v); A=D[tr]-D[tr].mean(0); yt=Y[tr]-Y[tr].mean(0); norm=np.maximum(np.linalg.norm(A,axis=0),1e-12); Q=np.empty((len(tr),0)); sel=[]; R=yt/np.std(yt,axis=0); R=R[:,[0,8]]
   for _ in range(k):
    B=A-Q@(Q.T@A); den=np.linalg.norm(B,axis=0); sc=np.linalg.norm(B.T@R,axis=1)/np.maximum(den,1e-8); sc[(den/norm)<1e-5]=0; j=np.argmax(sc); sel.append(j); qj=B[:,j]/den[j]; Q=np.c_[Q,qj]; R=R-np.outer(qj,qj@R)
   a=np.c_[np.ones(n),D[:,sel]]; h=a[v]@np.linalg.lstsq(a[tr],Y[tr],rcond=None)[0];es.append(err(h,v));picked.append(sel)
  print('OMP',name,k,report(es),'selected',picked[0] if k==15 else '')
