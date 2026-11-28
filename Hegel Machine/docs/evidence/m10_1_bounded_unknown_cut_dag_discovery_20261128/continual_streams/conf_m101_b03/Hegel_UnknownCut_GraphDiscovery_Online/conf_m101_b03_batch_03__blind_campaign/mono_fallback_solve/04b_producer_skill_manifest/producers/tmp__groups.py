import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=np.load('/tmp/explore.npz'); z,b,Y=q['z'],q['b'],q['Y']; n=52
p=s['panel_c'];u=p['u'];v=p['v1']; stack=np.column_stack

def design(x,h):
 a,f,r=h; return stack((np.ones(len(x)),np.tanh(a*x)+r*x/(1+(f*x)**2)))
def resid(h):
 D=design(u,h);return (D@np.linalg.lstsq(D,v,rcond=None)[0]-v).ravel()
r=min((least_squares(resid,h,max_nfev=180,bounds=([.2,.2,-30],[16,12,30])) for h in it.product((2,5,8),(1,4,7),(-4,2))),key=lambda r:r.cost); print('C1 pars',r.x,np.std(r.fun),flush=True)
W=np.linalg.lstsq(design(u,r.x),v,rcond=None)[0]; WD=np.linalg.lstsq(stack((np.ones(len(u)),np.cos(2.4*u))),p['v2'],rcond=None)[0]
for k in ('panel_a','panel_b','panel_c'):
 h=s[k];print(k,[(a,np.min(x,axis=0).round(2),np.max(x,axis=0).round(2)) for a,x in h.items()]);
print('z/b range',[(np.min(x,axis=0),np.max(x,axis=0)) for x in [z,b]],flush=True)
B=np.linalg.lstsq(stack((np.ones(len(s['panel_b']['u'])),s['panel_b']['u'])),s['panel_b']['v'],rcond=None)[0]
for cs,ds in it.product('zb',repeat=2):
 a={'z':z,'b':b}; c=design(a[cs],r.x)@W; d=stack((np.ones(len(z)),np.cos(2.4*a[ds])))@WD
 A={'z':z,'b':b,'c':c,'d':d}; G={'lin'+k:x for k,x in A.items() if k!='b'}
 for k,l in it.combinations_with_replacement(A,2):
  x,w=A[k],A[l]
  G[k+l+'dot']=(x*w).sum(1)[:,None];G[k+l+'diag']=x*w
  if k==l:
   if k!='b': G[k+l+'full']=stack([x[:,i]*x[:,j] for i,j in it.combinations_with_replacement(range(4),2)])
  else:
   for shift in (1,3):
    G[k+l+'r'+str(shift)]=x*np.roll(w,shift,axis=1)
   G[k+l+'full']=np.einsum('ni,nj->nij',x,w).reshape(len(x),16)
 G['z3']=z**3;G['b3']=b**3
 G['cyclic']=z*np.roll(z,1,axis=1);G['zcos']=np.cos(2.4*z);G['bcos']=np.cos(2.4*b)
 for k in ['z','b','c','d']:
  x=A[k];G[k+'normrat']=(x/(1+np.sum(x*x,axis=1,keepdims=True)));G[k+'rat']=x/(1+x*x)
 names=list(G); D=stack((np.ones(len(z)),*G.values())); ends=np.cumsum([1]+[x.shape[1] for x in G.values()]); yy=Y[:,[0,1,4,8]]; zz=yy[:n]-yy[:n].mean(0); sy=zz.std(0)
 def beam(targets):
  seen={}; ix=[np.arange(i,j) for i,j in zip(ends[:-1],ends[1:])]; pool=[()]; yt=yy[:,targets]; norm=sy[targets]
  for step in range(8):
   new=[]
   for ks in pool:
    for f in range(len(names)):
     if f in ks:continue
     key=tuple(sorted((*ks,f)))
     if key in seen:continue
     cols=np.concatenate(([0],*(ix[j] for j in key))); E=D[:,cols]
     if len(cols)>46:continue
     U,S,V=np.linalg.svd(E[:n],full_matrices=False); k=S>S[0]*1e-9; U=U[:,k]; co=U.T@yt[:n]; err=(yt[:n]-U@co)/np.maximum(1-(U*U).sum(1,keepdims=True),1e-5); score=np.mean(np.abs(err/norm)); seen[key]=score
     new.append((score,key,np.mean(abs(E[n:]@((V[k].T/S[k])@co)-yt[n:]),axis=0)))
   if not new:break
   new.sort(key=lambda x:x[0]);pool=[a[1] for a in new[:14]]
   a=new[0]; print(cs+ds,targets,step,round(a[0],6),a[2].round(6),[names[j] for j in a[1]],flush=True)
 for targets in [[0],[1,2,3]]:beam(targets)
