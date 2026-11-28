from probe_walls import *
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
def rm(R): return np.sqrt(np.mean(R**2,axis=0))
print('reflect residual by shift', np.vstack([rm(r) for r in np.split(resid([1,1,0,0]*2),len(s['mu_shifts']))]))
print('reflect residual after nuisance',rm((np.eye(len(Z))-Q@Q.T)@resid([1,1,0,0]*2)))
print('reflect inferred pulse nuisance', np.linalg.lstsq(Z,-resid([1,1,0,0]*2),rcond=None)[0])
def gen(q, variant='wrap'):
 out=[]
 for k,ab in enumerate('xy'):
  kap=s['kappa_'+ab];vel=s['vel_'+ab];n=len(kap);h=s['L'+ab]/(n-1);A=old._axis(kap,vel,h).tolil()
  for j,(i,inward,opp) in enumerate(((0,1,n-1),(n-1,n-2,0))):
   a,b,c=q[6*k+3*j:6*k+3*j+3];local=a*kap[i]/h**2;leak=b*s['robin_'+ab]/h;extra=c*kap[i]/h**2
   A[i,i]=local+leak+extra;A[i,inward]=-local
   if variant=='wrap': A[i,opp]=-extra
   elif variant=='second': A[i,2*inward-i]=-extra
   elif variant=='mirror': A[i,n-1-inward]=-extra
  out.append(A.tocsc())
 ax,ay=out;return sparse.kron(sparse.eye(s['ny']),ax,format='csc')+sparse.kron(ay,sparse.eye(s['nx']),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
R=np.eye(len(Z))-Q@Q.T
for variant in ('base','wrap','second','mirror'):
 for use_z in (False,True):
  def f(q):
   if variant=='base': a=op(q)
   else: a=gen(q, variant)
   e=old._forward(s,a)[0]@W-Y
   return R@e if use_z else e
  for omit in (-1,0,1,2):
   keep=np.arange(3)!=omit
   x0=([1,1,.05,.05]*2 if variant=='base' else [1,.05,.05]*4)
   opt=least_squares(lambda q:f(q)[:,keep].ravel(),x0,bounds=(0,8),max_nfev=100,gtol=1e-12)
   print(variant,use_z,omit,np.round(opt.x,5),rm(f(opt.x)))
