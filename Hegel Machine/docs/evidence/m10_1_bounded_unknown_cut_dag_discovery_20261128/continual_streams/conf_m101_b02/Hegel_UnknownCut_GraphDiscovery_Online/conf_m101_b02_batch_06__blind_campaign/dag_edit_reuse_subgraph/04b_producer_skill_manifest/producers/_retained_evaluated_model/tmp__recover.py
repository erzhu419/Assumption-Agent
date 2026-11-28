from probe import *
from scipy.linalg import qr
from scipy.spatial.distance import cdist
from sklearn.linear_model import ElasticNet, ARDRegression
from itertools import combinations_with_replacement as comb

def ls(a,y):return np.linalg.lstsq(a,y,rcond=1e-9)[0]
def one(a):return np.c_[np.ones(len(a)),a]
def cyc(a):return a*np.roll(a,-1,axis=1)
w1=ls(one(cyc(u)),v1); w2=ls(one(np.c_[u,u*u]),v2)
def c(a):return one(cyc(a))@w1
def d(a):return one(np.c_[a,a*a])@w2
x=s['train']['x']; n=len(x); Y=np.c_[y,tr]; order=np.random.default_rng(54).permutation(n); folds=[np.arange(int(.75*n),n),*np.array_split(order,4)]
def errors(pred,j):
 r=abs(pred-Y[j]);return [r[:,0].mean(),r[:,1:].mean()]
def report(tag,es):print(tag,np.round(es[0],5),np.round(np.max(es[1:],axis=0),5))
def withfold(M,fn):
 es=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j);es.append(errors(fn(M,i,j),j))
 return es
H=lambda k: np.c_[np.ones(len(t)),-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)]
k=least_squares(lambda k:(H(k)@ls(H(k),tr.T)-tr.T).ravel(),[.38,1.56],bounds=([.05,.6],[1,5])).x
print('coefficient fit and orthogonal time'); Q=np.linalg.qr(H(k)[:,::-1])[0]; print(Q.T); print('trajectory readout residual exact rank3',np.std(tr-tr@Q@Q.T));
core={'x':x,'z':z,'b':b,'cx':c(x),'cz':c(z),'cb':c(b),'dx':d(x),'dz':d(z),'db':d(b)}
for tag in ('x,z,b,cx','x,z,b,cx,cz,cb','x,z,b,cx,cz,cb,dx,dz,db','z,b,cz,cb,dz,db','x,cx','x,z,cx,cz','x,b,cx,cb'):
 data=[core[k] for k in tag.split(',')]; M=one(np.column_stack(data)); report(('linear',tag),withfold(M,lambda a,i,j:a[j]@ls(a[i],Y[i])))
for tag in ['x,z,b','x,z,b,cx,cz,cb','x,z,b,cx,cz,cb,dx,dz,db','z,b,cz,cb,dz,db']:
 parts=tag.split(','); bases=[core[k] for k in parts]; mats={}; names=[]
 for k,F in zip(parts,bases):
  for lab,g in [('lin',lambda a:a),('sq',np.square),('cube',lambda a:a**3),('tanh',lambda a:np.tanh(2*a)),('sin',lambda a:np.sin(2*a)),('rat',lambda a:a/(1+3*a*a))]:
   M=g(F); mats[k+lab]=M; names.extend([f'{lab}({k}{j})' for j in range(4)])
 for p,a in enumerate(bases):
  for q,b0 in enumerate(bases[:p+1]):
   if p>=3 and q>=3:continue
   ix=[(j,l) for j in range(4) for l in range(4) if p!=q or j<l]; mats[f'{p}_{q}']=np.column_stack([a[:,j]*b0[:,l] for j,l in ix]);names.extend([f'{parts[p]}{j}*{parts[q]}{l}' for j,l in ix])
 M=np.column_stack(list(mats.values())); YY=np.c_[y,tr@Q]; rec=lambda L:np.c_[L[:,0],L[:,1:]@Q.T]
 for al in (.0003,.001,.003,.01):
  def model(a,i,j):
   mu=a[i].mean(0); sd=a[i].std(0).clip(1e-8); aa=(a-mu)/sd; ss=YY[i].std(0); out=[]
   for col,val in enumerate((YY[i]/ss).T):
    ef=ElasticNet(alpha=al,l1_ratio=.95,max_iter=6000,tol=2e-5).fit(aa[i],val);out.append(ef.predict(aa[j])*ss[col]);
    if j is folds[0] and al==.003 and col<2: print('largest',tag,col,[(names[l],round(ef.coef_[l],3)) for l in np.argsort(-abs(ef.coef_))[:10]])
   return rec(np.column_stack(out))
  report(('sparse',tag,al),withfold(M,model))
 for power in (2,3):
  for within in (.15,1.,4.):
   FF=[(F-F.mean(0))/F.std(0) for F in bases]; L=[F@F.T/4 for F in FF]; K=(1+sum(L)/len(L))**power+within*sum((1+l)**power for l in L)/len(L)
   for la in (.00003,.001,.01):
    es=withfold(K,lambda K,i,j:K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+la*np.eye(len(i)),Y[i]));
    if max(es[0])<.07:report(('kernel',tag,power,within,la),es)
