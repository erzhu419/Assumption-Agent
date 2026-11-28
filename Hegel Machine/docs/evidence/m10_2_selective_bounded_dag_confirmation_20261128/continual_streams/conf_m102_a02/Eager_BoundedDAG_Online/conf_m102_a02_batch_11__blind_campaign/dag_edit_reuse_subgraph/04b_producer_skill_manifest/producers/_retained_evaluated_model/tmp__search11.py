exec(open('/tmp/probe11.py').read().split('for k,p in')[0])
import sys
sys.path.insert(0,'/holdout_task');import node_01 as m
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
z=m._latent(s['panel_a'],x,None); h=s['panel_b']['u']; v=s['panel_b']['v'];p=s['panel_c'];w1,w2=s['telemetry_vec'][:2]
fs={'sin':np.sin,'tanh':np.tanh,'ash':np.arcsinh,'atan':np.arctan,'erf':erf,'exp':np.exp,'cos':np.cos,'rat':lambda u:u/(1+u*u),'sqt':lambda u:np.sqrt(1+u*u)-1,'soft':lambda u:u/np.sqrt(1+u*u)}
def pd(u,k,w):return np.c_[aff(u),fs[k](w*u)]
opts=[]
for k in fs:
 def loss(w):
  a=pd(h,k,w);return np.mean((a@fit(a,v)-v)**2)
 o=minimize_scalar(loss,bounds=(.1,6),method='bounded');opts.append((o.fun,k,o.x));print('B',k,o.x,np.sqrt(o.fun),np.sqrt(loss(w1)),np.sqrt(loss(w2)))
_,k,w=min(opts);g=pd(z,k,w)@fit(pd(h,k,w),v);print('B range',g.min(),g.max()); print('B coeff',fit(pd(h,k,w),v).round(3));
q=pd(z,'ash',3.39)@fit(pd(p['u'],'ash',3.39),p['v1']); r=pd(z,'sqt',np.sqrt(.927))@fit(pd(p['u'],'sqt',np.sqrt(.927)),p['v2'])
sets={'z':z,'g':g,'q':q,'r':r}; groups={'a':z,'z2':m._basis(z,2)[:,5:],'z3':m._basis(z,3)[:,15:],'g':g,'q':q,'r':r};
for i,j in itertools.combinations_with_replacement(sets,2):
 a,b=sets[i],sets[j];groups[i+j]=(a[:,:,None]*b[:,None,:]).reshape(len(z),-1)
 for l in range(4):groups[i+j+str(l)]=a*np.roll(b,l,axis=1)
for k in fs:
 for w in (w1,w2):groups[k+str(w)]=fs[k](w*z)
for k,gp in list(groups.items()):
 for suf,b in [('2',z),('3',m._basis(z,2)[:,1:])]:
  if gp.shape[1]<=5:groups[k+'*'+suf]=(gp[:,:,None]*b[:,None,:]).reshape(len(z),-1)
inds=np.arange(len(x));rng=np.random.default_rng(22);folds=[(inds[:n],inds[n:])]+[(np.setdiff1d(inds,j),j) for j in np.array_split(rng.permutation(inds),5)]
def cv(a,lam):
 a=(a-a.mean(0))/np.maximum(a.std(0),1e-8); k=a@a.T
 return np.array([np.mean(np.abs(y[j]-y[i].mean()-(k[np.ix_(j,i)]-k[np.ix_(i,i)].mean(0))@solve(k[np.ix_(i,i)]+lam*np.eye(len(i)),y[i]-y[i].mean(),assume_a='pos'))) for i,j in folds])
base=[z,groups['z2'],g,q,r]; results=[]
for key,b in groups.items():
 for basis in [base,base[:2],[z,g,q,r]]:
  for lam in (.000003,.0003,.01,.2):
   sc=cv(np.column_stack((*basis,b)),lam);results.append((sc[1:].mean(),sc.max(),sc[0],key,len(basis),lam))
print('top OOF',sorted(results)[:16]);print('top public',sorted(results,key=lambda a:a[2])[:12]);
for kernel in ['all','plain']:
 k=np.zeros((len(z),len(z)))
 for a in ([z,g,q,r] if kernel=='all' else [z]):
  d=((a[:,None]-a[None,:])**2).sum(2)/np.mean(np.var(a,0));
  for scale in [.005,.03,.15]:k+=np.exp(-scale*d)
 for l in (1e-6,1e-4,.001):
  sc=[np.mean(abs(y[j]-k[np.ix_(j,i)]@solve(k[np.ix_(i,i)]+l*np.eye(len(i)),y[i],assume_a='pos'))) for i,j in folds];print('kernel',kernel,l,sc)
