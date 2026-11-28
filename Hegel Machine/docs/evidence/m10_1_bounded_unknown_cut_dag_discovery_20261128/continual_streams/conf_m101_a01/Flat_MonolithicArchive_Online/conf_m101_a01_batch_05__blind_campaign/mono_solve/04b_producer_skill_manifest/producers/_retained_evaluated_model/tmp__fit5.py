from explore import *
from itertools import combinations,product
from sklearn.linear_model import orthogonal_mp
from scipy.spatial.distance import cdist
np.set_printoptions(precision=3,suppress=True,linewidth=190)
folds=[np.arange(52,72),np.arange(56,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(4)]
def cv(K,target,regs):
 e=[]
 for reg in regs:
  r=[]
  for te in folds:
   tr=np.setdiff1d(np.arange(len(y)),te); p=K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+reg*np.eye(len(tr)),target[tr]);r.append(np.mean(abs(target[te]-p)))
  e.append([reg,*r[:2],np.mean(r[2:]),max(r)])
 return np.array(e)
spaces={'x':x,'d':d,'b':b,'z(d)':z(d),'z(b)':z(b),'w(d)':w(d),'w(b)':w(b),'h':d+(w(d)-B[0])@Bi}
for k,a in spaces.items():
 print('\n',k,'poly',ls(aug(old._poly(a,[1,2,3])),y))
for target in [y,T]:
 res=[]
 for k,a in list(spaces.items())+[(i+'+'+j,np.c_[spaces[i],spaces[j]]) for i,j in product(['d','b'],['z(d)','z(b)','w(d)','w(b)','h'])]:
  a=a-a.mean(0); a/=np.mean(a.std(0)); G=a@a.T/a.shape[1]; D=cdist(a,a,'sqeuclidean')/a.shape[1]
  for order in [2,3,4,5,6]:
   for gain in [.15,.5,1.5]:
    for j in cv((1+gain*G)**order,target,[.00001,.0001,.001,.01]):res.append((j[3],k,('poly',order,gain),j))
  for l in [.3,.6,1,2,3]:
   R=np.sqrt(5*D)/l; M=(1+R+R**2/3)*np.exp(-R); S=np.exp(-D/(2*l*l))
   for nm,K in [('m52',M),('rbf',S),('poly+m52',(1+G)**2+M),('poly+rbf',(1+G)**2+S)]:
    for j in cv(K,target,[.00001,.0001,.001,.01]):res.append((j[3],k,(nm,l),j))
 print('kernel',target.shape,*sorted(res,key=lambda r:r[0])[:13],sep='\n')

def sparse(spaces,target):
 names=[]; F=[]
 for k,a in spaces.items():
  dirs=[(str(i),a[:,i]) for i in range(4)]+[(f'{i}{sg:+d}{j}',a[:,i]+sg*a[:,j]) for i,j in combinations(range(4),2) for sg in [-1,1]]
  for nm,v in dirs:
   for nn,f in [('lin',v),('abs',abs(v)),('q',v*v),('cube',v**3),('4',v**4),('5',v**5)]+[(f'{nn}{g}',f) for g in [.5,1,1.7,2.5] for nn,f in [('sin',np.sin(g*v)),('cos',np.cos(g*v)),('tanh',np.tanh(g*v)),('rat',v/(1+g*v*v)),('rqt',v*v/(1+g*v*v)),('fat',v/(1+g*abs(v)))]]:
    F.append(f); names.append(k+nm+nn)
  for deg in [2,3,4]:
   for js in combinations_with_replacement(range(4),deg):
    F.append(np.prod(a[:,js],axis=1));names.append(k+str(js))
 for ka,kb in combinations(spaces,2):
  for i,j in product(range(4),repeat=2):
   F.append(spaces[ka][:,i]*spaces[kb][:,j]);names.append(f'{ka}{i}*{kb}{j}')
 F=np.array(F).T
 for ns in [3,5,8,12,16,20,25,30,35]:
  r=[]
  for te in folds:
   tr=np.setdiff1d(np.arange(len(y)),te); f=(F-F[tr].mean(0))/np.maximum(1e-8,F[tr].std(0)); yy=target[tr]-target[tr].mean(0); cc=orthogonal_mp(f[tr]/np.sqrt(len(tr)),yy,n_nonzero_coefs=ns); p=f[te]@cc/np.sqrt(len(tr))+target[tr].mean(0);r.append(np.mean(abs(target[te]-p)))
  print('omp',list(spaces),ns,np.round([*r[:2],np.mean(r[2:]),max(r)],4))
 f=(F-F.mean(0))/np.maximum(1e-8,F.std(0));cc=orthogonal_mp(f/np.sqrt(len(y)),target-target.mean(),n_nonzero_coefs=12); print('omp full',[(names[i],round(cc[i],3)) for i in np.flatnonzero(cc)])
for kk in [['d'],['b'],['d','b'],['d','b','z(d)','z(b)','w(d)','w(b)','h']]:sparse({k:spaces[k] for k in kk},y)
