exec(open('/tmp/compose.py').read().split('ops={}')[0])
from scipy.linalg import qr
from itertools import combinations_with_replacement,combinations
from sklearn.linear_model import OrthogonalMatchingPursuit
for k,key,fn in [('v','panel_b',FNS['erf']),('v1','panel_c',n.arcsinh)]:
 p=s[key];e,op,q,c=opfit(p['u'],p[k],fn,True);print(k,q,c); print('ratios',c[1:5]/c[5:]);globals()[k]=op
p=s['panel_c'];ds=lambda x:n.c_[n.ones(len(x)),x,x*x]; c2=ls(ds(p['u']),p['v2']); V=lambda x:ds(x)@c2; print('v2',c2)
print('summary',s['train']['summary'],'y range',y.min(),y.max())
B,C=v,v1
F=temporal(n.log([.41,1.795])); H=ls(F,T.T).T
print('stack corr',n.corrcoef(n.c_[y,H,z,B(z),C(z),V(z),C(B(z))].T)[:3]); print('y and mode values',n.c_[y,H][:22])
base={'z':z,'B':B(z),'C':C(z),'V':V(z),'CB':C(B(z)),'BC':B(C(z)),'VB':V(B(z)),'VC':V(C(z)),'BV':B(V(z)),'CV':C(V(z))}
# Test ridge-regularized multioperator feature maps, grouped by shared construction.
rng=n.random.default_rng(0); splits=[n.arange(72)]+[rng.permutation(72) for _ in range(6)];
def cv(K):
 R=[]
 for iy in splits:
  i,j=iy[:52],iy[52:]; lam=10.**n.arange(-6.,1.); d,U=n.linalg.eigh(K[n.ix_(i,i)]); p=(K[n.ix_(j,i)]@U)[None,:,:]@((U.T@Y[i])[None,:,:]/(d[None,:,None]+lam[:,None,None])); R.append(n.mean(abs(p-Y[j]),axis=1));
 R=n.array(R); return R[0,:,0],n.mean(R[1:,:,0],axis=0),R[0,:,1:].mean(1),n.mean(R[1:,:,1:],axis=(0,2))
def kernels(a):
 a=(a-a.mean(0))/n.maximum(a.std(0),.04); M=a@a.T/a.shape[1]; dist=n.maximum(0,n.diag(M)[:,None]+n.diag(M)[None,:]-2*M); D=n.abs(a[:,None,:]-a[None,:,:]);
 return {'linear':1+M,**{f'p{d}':(1+M)**d for d in [2,3,4]},**{f'g{c}':1+M+n.exp(-dist*c) for c in [.2,1]},**{f'a{c}':1+M+n.exp(-D**2*c).mean(-1) for c in [.5,2,7]}}
allk={k:kernels(a) for k,a in base.items()}; groups=[(k,) for k in base]+list(combinations(base,2))+[tuple(base),('z','B','C','V'),('z','CB','V'),('z','CB','B','C','V'),('z','CB','V','VC')]; rows=[]
for g in groups:
 for typ in allk['z']:
  K=sum(allk[k][typ] for k in g)/len(g); E=cv(K); rows.extend((g,typ,l,*e) for l,e in enumerate(zip(*E)))
for col in [3,4,5,6]:print('best',col,*sorted(rows,key=lambda r:r[col])[:6],sep='\n')
# Direct structured fits for scalar y using base features plus the observed temporal amplitudes to locate the cut.
for ky,w in list(base.items())+[('H',H)]:
 for d in [1,2,3]:
  D=poly(w,d);c=ls(D[:52],y[:52]);print('y fit',ky,d,n.mean(abs(D[52:]@c-y[52:])),n.std(D@c-y))
