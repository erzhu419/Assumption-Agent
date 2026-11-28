from combine import *
from scipy.optimize import least_squares
p=s['panel_b'];u=p['u'];v=p['v']; P,n=poly(u,3); ids=list(range(15))+[n.index(i) for i in ['000','001','011','111','222','223','233','333']];c=ls(P[:,ids],v);print('B polynomial',list(zip(np.array(n)[ids],np.round(c,4))))
for name,fn in {**families,'cubic':lambda x,k:x**3}.items():
 def design(x,k):return bias(np.c_[x,fn(x,np.asarray(k))]);
 def err(k):
  F=design(u,k);return (F@ls(F,v)-v).ravel()
 if name=='cubic': continue
 r=least_squares(err,[.9]*4,bounds=(.02,8)); print('B separate',name,np.round(r.x,4),np.std(r.fun))
# Determine whether B is a rotation of a separable saturation, or saturation after a rotation.
for j,pair in enumerate([(0,3),(0,3),(1,2),(1,2)]):
 x=u[:,pair];tar=v[:,j];out=[]
 for name,fn in families.items():
  def err(par):
   a,k,w=par; z=x@np.array([np.cos(a),np.sin(a)]); q=bias(np.c_[x,fn(z+w,k)]);return q@ls(q,tar)-tar
  for a in np.arange(0,np.pi,.48):
   r=least_squares(err,[a,1,0],bounds=([-8,.02,-1],[8,8,1]),max_nfev=100);out.append((np.std(r.fun),name,r.x))
 print('rotated B',j,sorted(out,key=lambda q:q[0])[:4])
tr=s['train'];x=tr['x'];T=tr['traj'];y=tr['y'];V=np.linalg.svd(T,full_matrices=False)[2]; a=T@V[0]*V[0,-1];D=get(s,x);
# Test compact interaction and projection features downstream rather than unrestricted panel combinations.
for k in ['z','zv','zvv1','zvv2']:
 z=D[k]; cols=[z[:,j] for j in range(4)]; labels=list('0123');
 for i,j in combinations(range(4),2):
  for sg in [-1,1]:cols.append((z[:,i]+sg*z[:,j])/np.sqrt(2)); labels.append(f'{i}{sg:+}{j}')
 Z=np.column_stack(cols);f=bias(z); nm=['1',*'0123'];
 for name,W in [('sq',Z**2),('cub',Z**3),('tanh',np.tanh(2*Z)),('sin',np.sin(3*Z)),('soft',Z/(1+abs(Z))),('sqrat',Z*Z/(1+2*Z*Z))]:f=np.c_[f,W];nm+= [name+h for h in labels]
 for i,j,l in combinations(range(4),3): f=np.c_[f,z[:,i]*z[:,j]*z[:,l]];nm+=[f'{i}{j}{l}']
 for tar in [y,a]:
  I=list(range(5)); R=list(range(5,f.shape[1])); g=f[:56];t=tar[:56]; line=[]
  for _ in range(8):
   j=min(R,key=lambda j:np.linalg.norm(t-g[:,I+[j]]@ls(g[:,I+[j]],t)));I.append(j);R.remove(j);c=ls(g[:,I],t);line.append((nm[j],round(np.std(t-g[:,I]@c),5),round(np.mean(abs(tar[56:]-f[56:,I]@c)),5)))
  print(k,line)
