exec(open('/tmp/probe.py').read().split('res=[]')[0])
from itertools import combinations_with_replacement
ij=list(combinations_with_replacement(range(4),2))
def quad(z): return np.c_[one(z),*[z[:,i]*z[:,j] for i,j in ij]]
C=lin(quad(c['u']),c['v1']); print('panel C coefficients',C.round(5)); q=lambda z:quad(z)@C
sc=[]
for k,z in zs.items():
 for name,w in [('q',q(z)),('qd',np.c_[q(z),d(z)]),('qz',np.c_[q(z),z]),('qdz',np.c_[q(z),d(z),z]),('quad',quad(z)[:,1:]),('dz',np.c_[d(z),z])]:
  for scale in [.35,.6,1.,1.5,2.]:
   e=cv(w,scale); sc.append((e.max(),e.mean(),k,name,scale,e.round(5).tolist()))
print(*sorted(sc)[:22],sep='\n'); print('best public',*sorted(sc,key=lambda r:max(r[-1][0]))[:8],sep='\n')
