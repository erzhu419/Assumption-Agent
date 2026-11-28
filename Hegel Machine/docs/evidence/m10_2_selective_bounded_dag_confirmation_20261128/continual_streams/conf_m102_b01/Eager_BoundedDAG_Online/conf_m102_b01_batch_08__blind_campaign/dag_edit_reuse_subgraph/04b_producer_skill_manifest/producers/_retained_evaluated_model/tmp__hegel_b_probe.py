exec(open('/tmp/hegel_panel_structure_probe.py').read().split('for name,x,v in')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=165)
X=b['u']; V=b['v']; print('B rows',np.c_[X,V]); print('C fits',ls(pol(c['u'],2),c['v1']).T,ls(pol(c['u'],2),c['v2']).T,sep='\n')
Z=np.concatenate([X,u]); A=pol(X,1); P=pol(Z,1)
mods={}
for name, fn in [('cube',lambda z,g:z+g*z**3),('rat',lambda z,g:z/(1+g*z*z)),('tanh',lambda z,g:np.tanh(g*z)),('atan',lambda z,g:np.arctan(g*z)),('sin',lambda z,g:np.sin(g*z)),('inv3',lambda z,g:np.sinh(np.arcsinh(g*z)/3)),('exp',lambda z,g:np.expm1(g*z))]:
 for ext in [0,1]:
  def mk(x,g):return np.c_[pol(x,1) if ext else np.ones(len(x)),fn(x,g)]
  k=minimize_scalar(lambda g:fits(mk(X,g),V)[1],bounds=(.01,6),method='bounded').x; d=mk(Z,k); rms,loo,q=fits(d[:len(X)],V); print('INPUT',name,ext,round(k,4),round(rms,6),round(loo,6));mods['in'+name+str(ext)]=d[len(X):]@q
 for ext in [0,1]:
  vals=[];errs=[]
  for v in V.T:
   def calc(p,d):
    z=d@p[:5]; return p[5]*fn(z,p[6])+p[7]+(d@p[8:] if ext else 0)
   p0=np.r_[ls(A,v),1.,1.,0.,np.zeros(5) if ext else []]
   sols=[]
   for g in [.3,1.,2.5]:
    p0[6]=g; ans=least_squares(lambda p:calc(p,A)-v,p0,max_nfev=350);sols.append((np.mean(ans.fun**2)**.5,ans.x))
   e,q=min(sols,key=lambda v:v[0]);errs.append(e);vals.append(calc(q,P[len(X):]))
  print('OUTPUT',name,ext,errs); mods['out'+name+str(ext)]=np.array(vals).T
np.savez('/tmp/hegel_b_probes.npz',**mods)
