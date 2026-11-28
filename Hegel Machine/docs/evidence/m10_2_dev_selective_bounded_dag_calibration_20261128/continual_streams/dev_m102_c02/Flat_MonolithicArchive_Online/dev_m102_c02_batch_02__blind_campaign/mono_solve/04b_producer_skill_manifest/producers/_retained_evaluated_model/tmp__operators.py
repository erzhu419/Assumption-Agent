exec(open('/tmp/explore.py').read().split('ops={}')[0])
from scipy.optimize import least_squares
from itertools import product
X=[('z',z)]; ys=[]
def fit(A,V):return lstsq(A,V)[0]
def add(na,D):
 for desc,A in [('lin',aff(D)),('bil',np.c_[aff(D),np.sum(D[:,:2]*D[:,2:],1)]),('pair',np.c_[aff(D),D[:,:2]*D[:,2:]]),('q',poly(D,2))]:
  ys.append((report(A,y)[1],report(A,y)[0],na,desc))
for pn in ['panel_b','panel_c']:
 p=s[pn];u=p['u'];
 for k,v in p.items():
  if k=='u':continue
  out=[]
  bases={**{f'poly{d}':lambda a,d=d:poly(a,d) for d in (1,2,3)},'pure_sq':lambda a:aff(a*a),'lin_sq':lambda a:aff(np.c_[a,a*a]),'lin_cube':lambda a:aff(np.c_[a,a**3]),'lin_cross':lambda a:aff(np.c_[a,a[:,np.triu_indices(4,1)[0]]*a[:,np.triu_indices(4,1)[1]]]),'lin_cubed_sq':lambda a:aff(np.c_[a,a*a,a**3]),'signed_sq':lambda a:aff(a*abs(a))}
  for tag,fn in {
    'cube':lambda a,h:a+h*a**3,'square':lambda a,h:a+h*a*a,'sat':lambda a,h:a/(1+h*a*a),'sqrt_sat':lambda a,h:a/np.sqrt(1+h*a*a),'tanh':lambda a,h:np.tanh(h*a),'sin':lambda a,h:np.sin(h*a),'sinh':lambda a,h:np.sinh(h*a),'power':lambda a,h:np.sign(a)*abs(a)**h,'exp':lambda a,h:np.expm1(h*a),'ratcube':lambda a,h:a**3/(1+h*a**4),'cubecross':lambda a,h:a**3+h*a}.items():
   best=None
   for ini in [.22,1.25]:
    def res(q):
     A=aff(fn(u,q[0]));return (A@fit(A,v)-v).ravel()
    lo=-.3 if tag in ['cube','square','sat','sqrt_sat','cubecross'] else .002
    r=least_squares(res,[ini],bounds=([lo],[5.]),max_nfev=100);score=np.std(r.fun)
    if best is None or score<best[0]:best=(score,r.x[0])
   par=best[1];bases[tag+':'+str(round(par,6))]=lambda a,fn=fn,par=par:aff(fn(a,par))
  for tag,f in bases.items():
   A=f(u);B=fit(A,v); pred=A@B; H=np.sum(A*lstsq(A,np.eye(len(A)))[0].T,1); rms=np.std(pred-v);loo=np.sqrt(np.mean(((pred-v)/np.maximum(1-H[:,None],1e-6))**2))
   name=pn[-1]+k[-1]+':'+tag
   out.append((rms,loo,name));D=f(z)@B;X.append((name,D));add(name,D)
   if tag in ['poly1','poly2','lin_sq','lin_cube']:print('coef',name,np.round(B,5))
  print('panel ranking', sorted(out)[:15])
print('scalar direct',*sorted(ys)[:25],sep='\n')
ys=[]
for (n1,A),(n2,B) in product(X,X):
 if n1[:2]!='c1' or n2[:2]!='c2':continue
 for n,F in [('sum(prod)',np.sum(A*B,1)[:,None]),('prod',A*B),('sum(shiftprod)',np.sum(A*np.roll(B,2,axis=1),1)[:,None]),('shiftprod',A*np.roll(B,2,axis=1)),('all', (A[:,:,None]*B[:,None,:]).reshape(len(A),-1))]:
  for L,V in [('z',z),('A',A),('B',B),('AB',np.c_[A,B])]:
   E=np.c_[aff(V),F];r=report(E,y);ys.append((r[1],r[0],n1,n2,n,L))
print('scalar combined',*sorted(ys)[:30],sep='\n')
