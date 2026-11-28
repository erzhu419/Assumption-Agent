import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/dag_sink_probe.py').read())
from scipy.spatial.distance import cdist
from scipy.linalg import solve
x=s['train']['x']; m=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); pa,pb=s['panel_a'],s['panel_b']; co,se=m['_fit_a'](pa['u'],pa['v']); ref=m['_invert_a'](x,co,se); kb,cb=m['_fit_b'](pb['u'],pb['v']); print('keys',s.keys(), 'panel C',c.keys(), 'A residual sorted',np.sort(np.linalg.norm(stack(*[m['_a_design'](pa['u'],j)@co[:,j] for j in range(4)])-pa['v'],axis=1))); print('B residual',np.std(m['_b_design'](pb['u'],kb)@cb-pb['v'],0)); print('C1 rotated tanh vs coordinate tanh'); from scipy.optimize import least_squares
for j in range(4):
 o=least_squares(lambda p: p[0]+p[1]*np.tanh(u@p[2:6]+p[6])+u@p[7:]-c['v1'][:,j],np.r_[0,1,ls(stack(one(u),u),c['v1'])[1:,j]*2,0,np.zeros(4)],max_nfev=500); print(j,np.std(o.fun),o.x)
D=stack(one(z),z,Q); R=Y-D@ls(D,Y); D0=np.c_[np.ones(len(z)),z]; V0=np.linalg.svd(tr-D0@ls(D0,tr),full_matrices=False)[2]; print('time modes after affine removed',V0[:3]); print('quadratic residual outliers (z, y, final, residual y & final, difference residual)',stack(z,y,tr[:,-1],R[:,0],R[:,-1],R[:,0]-R[:,-1])[np.argsort(abs(R[:,0]))[-18:]]); print('latent all ranges',np.min(z,0),np.max(z,0)); print('quad time ratios',ls(D,tr)[5:]/ls(D,tr)[5:,-1,None]);
folds=[(np.arange(52),np.arange(52,len(z)))]+[(np.flatnonzero(np.arange(len(z))%5!=i),np.flatnonzero(np.arange(len(z))%5==i)) for i in range(5)]
def evaluate(K):
 errors=[]
 for a,b in folds:
  e=np.abs(K[np.ix_(b,a)]@solve(K[np.ix_(a,a)],Y[a],assume_a='pos')-Y[b]); errors.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.round([*errors[0],*np.mean(errors[1:],0),*np.max(errors[1:],0)],5)
for nm,F in [('z',z),('ab',stack(A,B)),('zab',stack(z,A,B)),('ref',ref)]:
 F=F/F.std(0); S=cdist(F,F,'sqeuclidean')/F.shape[1]; P=(F@F.T)/F.shape[1]; G=(D/D.std(0,where=(np.arange(D.shape[1])!=0)[None,:],keepdims=True)) if False else D/np.r_[1,D[:,1:].std(0)]; G=G@G.T
 out=[]
 for gamma in [.03,.1,.3,1,3]:
  for lam in [.00003,.0003,.003,.03,.3]:
   out.append((evaluate(G+np.exp(-gamma*S)+lam*np.eye(len(z))),gamma,lam))
 print(nm,'kernel heldout and mean/worst CV', sorted(out,key=lambda p:max(p[0][2:4]))[:4]);
features={'signed_sq':z*abs(z),'cube':z**3,'pow4':z**4,'abs':abs(z),'tanh':np.tanh(1.7*z),'abs_pair':stack(*[abs(z[:,i]+sg*z[:,j]) for i,j in combinations(range(4),2) for sg in [-1,1]]),'signed_pair':stack(*[(p:=z[:,i]+sg*z[:,j])*abs(p) for i,j in combinations(range(4),2) for sg in [-1,1]]),'rat_pair':stack(*[z[:,i]/(1+g*z[:,j]**2) for i in range(4) for j in range(4) for g in [.3,1,3]]),'triple':stack(*[z[:,i]*z[:,j]*z[:,k] for i,j,k in combinations(range(4),3)]),'panel_power':stack(A*A,B*B,A*B),'ratio_q':stack(*[Q[:,j]/(1+g*z[:,k]**2) for j in range(10) for k in range(4) for g in [.3,1,3]])}
G=D/np.r_[1,D[:,1:].std(0)]; G=G@G.T
for nm,F in features.items():
 F=(F-F.mean(0))/F.std(0); K=F@F.T/F.shape[1]; print(nm,[evaluate(G+K+lam*np.eye(len(z))) for lam in [.0001,.003,.03]])
