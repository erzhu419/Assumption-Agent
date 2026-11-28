import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/dag_sink_probe.py').read())
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
m=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); pa=s['panel_a']; co,se=m['_fit_a'](pa['u'],pa['v']); ref=m['_invert_a'](s['train']['x'],co,se); raw=s['train']['x']; n=len(y); ii=np.arange(n); splits=[ii>=52]+[ii%5==i for i in range(5)]
ca=ls(t1(u,k,True),c['v1']); cb=ls(t2(u),c['v2']); ca0=ls(t1(u,tel[0],False),c['v1']);
def q(x): return stack(*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)])
def lin(x):return stack(one(x),x)
# Estimate shared response after removing affine terms; compare against an unconstrained time model.
for W in [z,ref,stack(z,ref)]:
 L=lin(W); R=Y-L@ls(L,Y); print('shared after affine',np.linalg.svd(R,full_matrices=False)[1][:4],ls(np.sqrt(t)[:,None],R[:,1:].T)@R[:,0]/(R[:,0]@R[:,0]));
T=np.sqrt(t); AA=tr@T/(T@T);
for L in [lin(z),lin(ref),lin(stack(z,ref))]:
 R=stack(AA,y)-L@ls(L,stack(AA,y)); print('amp/y slope and covariance',ls(R[:,:1],R[:,1]),np.corrcoef(R.T)[0,1],np.std(R,axis=0));
for name,tf in [('log',lambda p:np.log1p(p*t)),('exp',lambda p:-np.expm1(-p*t)),('lin',lambda p:t),('rat',lambda p:t/(1+p*t))]:
 opt=least_squares(lambda p:(tr-tr@np.linalg.pinv(stack(T,tf(p[0])).T)@stack(T,tf(p[0])).T).ravel(),[1.],bounds=(.00001,10)); print('fixed sqrt +',name,opt.x,np.std(opt.fun))
D={};
for name,W in [('z',z),('ref',ref),('raw',raw)]:
 P=t1(W,k,True)@ca; P0=t1(W,tel[0],False)@ca0; S=t2(W)@cb
 for nm,F in [('q',q(W)),('A',P),('B',S),('AB',stack(P,S)),('Aq',q(P)),('Bq',q(S)),('ABq',q(stack(P,S))),('AB*',P*S),('same',stack(P*S,P*W,S*W)),('quart',q(W**2)),('tanhq',q(np.tanh(tel[0]*W))),('sigq',q(W/np.sqrt(1+W*W))),('logcos',np.logaddexp(W,-W)),('cos',np.cos(tel[0]*W))]: D[name+'-'+nm]=F
 if name!='raw':
  for g in [0.2,0.5,1,1.7,3]:
   for fn,f in [('tan',np.tanh),('sin',np.sin),('exp',lambda a:a*np.exp(-a*a))]:
    J=f(g*W); D[name+fn+str(g)]=stack(J,q(J)); D[name+fn+'mix'+str(g)]=stack(J,q(stack(W,J)))
D['ref-zq']=q(stack(ref,z)); D['q-ref+z']=stack(q(ref),q(z)); D['joint-branches']=q(stack(A,B,t1(ref,k,True)@ca,t2(ref)@cb))
base=lin(stack(z,ref));
def err(F, mode, alpha):
 out=[]
 for h in splits:
  Z=base; G=F/np.maximum(F[~h].std(0),.001); G1=G[~h]; Z1=Z[~h]; C=ls(Z1,G1); G=G-Z@C; Ga=G[~h]; Zco=ls(Z1,Y[~h]); resid=Y[~h]-Z1@Zco
  if mode:
   U,S,V=np.linalg.svd(resid,full_matrices=False); v=V[0]; targets=resid@v
  else: targets=resid
  w=solve(Ga.T@Ga+alpha*np.eye(Ga.shape[1]),Ga.T@targets,assume_a='pos'); pred=G[h]@w
  if mode: pred=pred[:,None]*v
  E=abs(Z[h]@Zco+pred-Y[h]); out.append([E[:,0].mean(),E[:,1:].mean()])
 return np.round(np.r_[out[0],np.mean(out[1:],0),np.max(out[1:],0)],5)
O=[]
for nm,F in D.items():
 for mode in (0,1):
  for al in (.0001,.01,.25,2): O.append((err(F,mode,al),nm,mode,al))
for idx in (0,2,4):
 print('rank by metric',idx)
 for v in sorted(O,key=lambda p:p[0][idx])[:8]:print(v)
