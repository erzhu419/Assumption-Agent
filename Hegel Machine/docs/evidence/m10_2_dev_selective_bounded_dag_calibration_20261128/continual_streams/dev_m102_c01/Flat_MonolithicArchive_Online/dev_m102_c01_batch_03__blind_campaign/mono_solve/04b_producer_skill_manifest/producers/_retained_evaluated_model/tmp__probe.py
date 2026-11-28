import pickle, numpy as np
from scipy.optimize import minimize_scalar
from arc import _calibration
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=map(tr.get,('x','y','traj')); n=52
np.set_printoptions(precision=4,suppress=True)
a,b=_calibration(s['panel_a']); print('cal',a,b); z=(x-b)/a
p=s['panel_b']; M=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; print('B gram',M[1:]@M[1:].T)
B=lambda u:M[0]+u@M[1:]; Bi=lambda u:(u-M[0])@np.linalg.inv(M[1:]);
fs={'linear':lambda u,k:u,'tanh':lambda u,k:np.tanh(k*u),'sin':lambda u,k:np.sin(k*u),'rat':lambda u,k:u/(1+k*u*u),'rat2':lambda u,k:np.c_[u,u*u/(1+k*u*u)],'abs':lambda u,k:np.c_[u,abs(u)],'quadratic':lambda u,k:np.c_[u,u*u],'root':lambda u,k:u/np.sqrt(1+k*u*u),'soft':lambda u,k:u/(1+k*abs(u)),'tanh2':lambda u,k:np.c_[u,np.tanh(k*u)],'cos':lambda u,k:np.c_[u,np.cos(k*u)]}
C=[]; p=s['panel_c']; u=p['u'];
for key in ['v1','v2']:
 v=p[key]; cand=[]
 for name,f in fs.items():
  def loss(k,ret=False):
   D=np.c_[np.ones(len(u)),f(u,k)]; W=np.linalg.lstsq(D,v,rcond=None)[0]; r=v-D@W; return (W,np.sqrt(np.mean(r*r))) if ret else np.mean(r*r)
  p0=minimize_scalar(loss,bounds=(.03,4),method='bounded').x; w,e=loss(p0,True); cand.append((e,name,p0,w))
 cand.sort(key=lambda p:p[0]); print(key,[(a,b,c) for a,b,c,d in cand]); e,key,k,w=cand[0]; C.append(lambda x,f=fs[key],k=k,w=w:np.c_[np.ones(len(x)),f(x,k)]@w)
U,sv,V=np.linalg.svd(t,full_matrices=False); T=t@V[:2].T; target=np.c_[y,T]; print('temporal',sv,V[:2]); print('target corr',np.corrcoef(target.T));
coords={'X':x,'Z':z,'A':x*a+b,'BX':B(x),'BZ':B(z),'BiX':Bi(x),'BiZ':Bi(z)}
for i,f in enumerate(C):
 for key,v in list(coords.items())[:7]:coords['C'+str(i)+key]=f(v)
rows=[]
for name,q in coords.items():
 for fkey,f in fs.items():
  for k in ([1] if fkey in ('linear','abs','quadratic') else [.1,.25,.5,.8,1.25,2,3]):
   F=f(q,k); i,j=np.triu_indices(F.shape[1]);
   for deg in (1,2):
    if deg==2 and F.shape[1]>4:continue
    D=np.c_[np.ones(len(q)),F] if deg==1 else np.c_[np.ones(len(q)),F,F[:,i]*F[:,j]];
    W=np.linalg.lstsq(D[:n],target[:n],rcond=None)[0]; r=D@W-target; e=np.mean(abs(r[n:]),axis=0); et=np.mean(abs(r[:n]),axis=0); rows.append((e,et,name,fkey,k,deg))
for i in range(3):
 print('BEST',i); print(*sorted(rows,key=lambda r:r[0][i])[:18],sep='\n')
