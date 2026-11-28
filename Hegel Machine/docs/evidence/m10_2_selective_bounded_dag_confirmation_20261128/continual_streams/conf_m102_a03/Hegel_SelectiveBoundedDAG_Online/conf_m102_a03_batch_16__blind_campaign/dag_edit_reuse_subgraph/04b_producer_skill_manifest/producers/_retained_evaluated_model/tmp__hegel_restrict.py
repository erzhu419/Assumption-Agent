exec(open('/tmp/hegel_modes.py').read().split('for name,D in {')[0]); exec(open('/tmp/hegel_modes.py').read().split('def reg(')[1].split('rr=[]')[0].join(['def reg(','']))
from scipy.linalg import solve
# Reconstruct the nearly affine underlying branch coordinates in the frozen basis.
W=np.linalg.lstsq(aff(b),w,rcond=None)[0]
wc=aff(b)@W
print('w affine residual',np.sqrt(np.mean((wc-w)**2)),np.max(abs(wc-w)),flush=True)
i,j=np.triu_indices(4); n=X.shape[1]; J=np.eye(n); L=np.r_[J[4:5],J[25:29]].T@W
L2=J[4:5].T*(W[:1]**2)+J[25:29].T@(2*W[:1]*W[1:])+J[15:25].T@((1+(i!=j))[:,None]*W[1:][i]*W[1:][j]);
print('derivation',abs(X@L-wc).max(),abs(X@L2-wc**2).max(), flush=True)
for k,F in {'diag':np.c_[J[:5].T,J[[5,9,12,14]].T,L2,L], 'nozquad':np.c_[J[:5].T,L2,L], 'Bfull':np.c_[J[:5].T,J[15:25].T,L], 'diagB':np.c_[J[:5].T,L2,L,J[5:15].T], 'orig':J.T}.items():
 res=[]
 for qreg in (.0001,.01,.04,.1,.25,1):
  for tau in (.005,.01,.02,.035,.055,.08,.12):
   for tt in (T,V):
    D=X@F; e=scores(D,np.full(D.shape[1],qreg),tau,tt); res.append((e[0]+e[1:].mean()+.3*max(e),qreg,tau,tt.shape[1],np.round(e,5)))
 print(k,sorted(res,key=lambda r:r[0])[:4], 'public best', min(res,key=lambda r:r[-1][0]),flush=True)
print('frozen output');print(open('/holdout_task/_frozen_nodes/node_00.py').read().split('def run_node')[1])
