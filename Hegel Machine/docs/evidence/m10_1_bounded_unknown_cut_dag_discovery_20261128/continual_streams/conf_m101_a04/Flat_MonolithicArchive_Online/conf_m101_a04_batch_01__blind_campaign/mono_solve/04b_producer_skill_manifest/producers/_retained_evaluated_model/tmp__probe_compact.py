from probe_graph import *
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=145)
p=S['panel_c']; u=p['u']; v=p['v2']; raw=ops(S,T['x']); x=raw['x']
def b(a,w):return np.c_[poly(a,1),np.cos(a*w)]
def residual(w):
 f=b(u,w);return (f@ls(f,v)-v).ravel()
w=least_squares(residual,np.full(4,1.29),bounds=(.6,2)).x
print('cos frequencies',w,'RMS shared/individual',np.std(residual(np.full(4,1.29140845))),np.std(residual(w)))
print('cos coefficients',ls(b(u,w),v))
V=lambda q:np.c_[np.ones(len(q)),q[:,0]*q[:,1],q[:,0]*q[:,3],q[:,1]*q[:,2],q[:,2]*q[:,3]]
c=V(x)@ls(V(u),p['v1']); print('compact first panel RMS',np.std(V(u)@ls(V(u),p['v1'])-p['v1']))
d=b(x,w)@ls(b(u,w),v)
L=np.c_[np.ones(len(c)),c,d,c*d]; H=np.c_[L,c[:,0]*d[:,1]]; F=np.c_[L,*[c[:,i]*d[:,j] for i in range(4) for j in range(4) if i!=j]]
print('diag + extra coefficients',ls(H[:52],Y[:52]));print('all cross coefficients',ls(F[:52],Y[:52])[-12:]);
splits=[(np.arange(52),np.arange(52,72))]+[(q[:56],q[56:]) for q in [np.random.default_rng(k).permutation(72) for k in range(12)]]
for j in range(4):
 q=np.argsort(x[:,j]);splits += [(q[16:],q[:16]),(q[:-16],q[-16:])]
for name,G,J in [('diag only',L,L),('diag plus extra',L,H),('diag plus extra both',H,H),('all cross',F,F)]:
 errs=[]
 for a,b_ in splits:
  p=np.c_[G[b_]@ls(G[a],Y[a,0]),J[b_]@ls(J[a],Y[a,1:])]; e=abs(p-Y[b_]);errs.append([np.mean(e[:,0]),np.mean(e[:,1:]),np.max(e[:,1:].mean(0))]);
 print(name,'prescribed',errs[0],'worst random',np.max(errs[1:13],0),'worst tails',np.max(errs[13:],0))
