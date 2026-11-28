from probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from itertools import product
np.set_printoptions(precision=4,suppress=True)
print('times',t,'telemetry',s['telemetry_vec'],'summary',tr['summary'],'observed',np.mean(y),np.std(y),np.mean(z[:,-1]),np.std(z[:,-1]));print('A',A,'B',B)
c=s['panel_c']; w=c['u'];
def form(u,k,p):return np.c_[aff(u),np.tanh(p*u)] if k==0 else np.c_[aff(u),u*u,np.cos(p*u)]
ops=[]
for k,target in enumerate((c['v1'],c['v2'])):
 def err(p):
  f=form(w,k,p);return (f@fit(f,target)-target).ravel()
 p=least_squares(err,[2.3 if k==0 else 2.5],bounds=(.2,5)).x;bb=fit(form(w,k,p),target);print('operator',k,p,bb);ops.append(lambda u,k=k,p=p,bb=bb:form(u,k,p)@bb)
g,h=ops
rot=[('raw',x),('cal',q),('afwd',aff(x)@A),('rot',r),('rot0',q@B[1:]),('rotx',aff(x)@B),('inv',(q-B[0])@np.linalg.inv(B[1:])),('rotT',q@B[1:].T),('atrot',aff(aff(x)@A)@B)]
Y=np.c_[y,z]; splits=[np.arange(56,72),*np.array_split(np.random.default_rng(18).permutation(72),4),np.argsort(np.linalg.norm(x,axis=1))[-16:]]
def design(d):
 fs=[]
 for name,u in rot:
  gu,hu=g(u),h(u)
  for tag,a in [('u',u),('g',gu),('h',hu),('h(g)',h(gu)),('g(h)',g(hu)),('g B',aff(gu)@B),('g postB',g(aff(u)@B)),('h postB',h(aff(u)@B))]:
   if d=='simple': yield name+':'+tag,poly(a,2);continue
   for vtag,b in [('u',u),('g',gu),('h',hu)]:
    if vtag==tag:continue
    yield name+':'+tag+'/'+vtag,np.c_[aff(u),a,b,a*b,*[a[:,i,None]*b for i in range(4)]],a,b

def test(name,K):
 E=[]
 for ids in splits:
  it=np.setdiff1d(np.arange(len(x)),ids);L=K[np.ix_(it,it)];L.flat[::len(it)+1]+=1e-5;P=K[np.ix_(ids,it)]@cho_solve(cho_factor(L),Y[it]);e=abs(P-Y[ids]);E.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
 return np.array(E)
R=[]
def save(name,K):
 E=test(name,K);R.append((name,E[0],np.mean(E,0),np.max(E,0)))
for name,f in design('simple'):
 save(name,f@f.T)
for name,f,a,b in design('full'):
 for lam in (.05,.5,2.):
  P=poly(q,2);K=P@P.T+lam*(f@f.T);save(name+' '+str(lam),K)
for name,u in rot:
 P=poly(u,2)
 for ix,k in enumerate((u,g(u),h(u),np.c_[g(u),h(u)],np.c_[u,g(u),h(u)])):
  k=k/np.std(k,axis=0); dd=cdist(k,k,'sqeuclidean')/k.shape[1];
  for scale,amp in product((.12,.5,1.3),(.03,.3)):
   save(name+' rbf'+str((ix,scale,amp)),P@P.T+amp*np.exp(-scale*dd))
for j in range(3):
 print('rank for',j)
 for row in sorted(R,key=lambda row:row[2][j]+.6*row[3][j])[:16]:print(row)
