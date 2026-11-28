from scalar_probe import *
from scipy.linalg import solve

def grid(tr,va):
 p,state,y,q,t,x,xq=fold(tr,va); a,b=p['scores'],state@p['modes'].T; sy,qy=p['sy'],p['qy']; st,qt=p['st'],p['qt']; m,r=len(tr),len(va); results=[]
 for label,f,g in [('sy',sy,qy),('st',st,qt),('x',x,xq),('both',np.c_[sy,st],np.c_[qy,qt]),('state',a,b),('sy+st+state',np.c_[sy,st,a],np.c_[qy,qt,b]),('sy+state',np.c_[sy,a],np.c_[qy,b]),('x+state',np.c_[x,a],np.c_[xq,b])]:
  for degree in [1,2,3,4,5,6]:
   scale=np.sqrt(np.mean(f*f,axis=0)); u=f/scale; v=g/scale; G=u@u.T/u.shape[1]; Gq=v@u.T/u.shape[1]; D=np.maximum(np.diag(G)[:,None]+np.diag(G)[None,:]-2*G,0); Dq=np.maximum(np.mean(v*v,axis=1)[:,None]+np.diag(G)[None,:]-2*Gq,0)
   for kind in ['poly','rbf']:
    H,Hq=((1+G)**degree,(1+Gq)**degree) if kind=='poly' else (np.exp(-D*degree/12),np.exp(-Dq*degree/12))
    for base in [0,1]:
     for ridge in [1e-6,1e-4,.001,.01,.1]:
      mat=H+base*p['K']+ridge*np.eye(m); c=solve(mat,np.c_[y-y.mean(),np.eye(m)],assume_a='pos'); pred=(Hq+base*p['Kq'])@c[:,0]+y.mean(); loo=np.abs(c[:,0]/np.diag(c[:,1:])).mean(); cv=np.abs(q-pred).mean(); results.append((cv,loo,label,degree,kind,base,ridge))
 print('best heldout',sorted(results)[:8]); print('best inner',sorted(results,key=lambda x:x[1])[:8]); print('best per group',[(k,min(u for u in results if u[2]==k)[:2]) for k in sorted(set(u[2] for u in results))]);
for tr,va in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]:grid(tr,va)
p,state,y,q,t,x,xq=fold(np.arange(52),np.arange(52,72)); print('ranges'); print('sy',p['sy'].min(0),p['sy'].max(0)); print('st',p['st'][:,:4].min(0),p['st'][:,:4].max(0)); print('singvals',np.linalg.svd(s['train']['traj'],compute_uv=False)); print('corr y raw and state',np.corrcoef(np.c_[y,x,p['scores']].T)[0]);
