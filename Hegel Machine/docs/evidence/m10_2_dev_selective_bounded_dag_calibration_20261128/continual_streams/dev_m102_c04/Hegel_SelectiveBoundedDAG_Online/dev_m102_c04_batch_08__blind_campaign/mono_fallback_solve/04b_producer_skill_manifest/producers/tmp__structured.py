import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
pa=s['panel_a']; u,v=pa['u'],pa['v']; m=np.linalg.norm(u-v,axis=1)<1; w=np.array([ls(aff(u[m])[:,[0,j+1]],v[m,j]) for j in range(4)]).T
pb=s['panel_b']; wb=ls(aff(pb['u']),pb['v']); B=lambda x:aff(x)@wb
pairs=[(0,1),(0,3),(1,2),(2,3)]; prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cos=lambda x:np.cos(2.4*x)
pc=s['panel_c']; wc1=ls(aff(prod(pc['u'])),pc['v1']); wc2=ls(aff(cos(pc['u'])),pc['v2']); C=lambda x:aff(prod(x))@wc1; D=lambda x:aff(cos(x))@wc2
x=s['train']['x']; z=x*w[1]+w[0]; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; tr=Y[:,1:];
def temporal(p):
 return -np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(temporal(p)@ls(temporal(p),tr[:52].T)-tr[:52].T).ravel(),[.5,2.5],bounds=(.01,30)); print('best rates',o.x); print('B',wb,'C',wc1,'D',wc2,sep='\n');
base={'x':x,'z':z,'inv':(x-w[0])/w[1]}; sets=dict(base)
for nm,a in base.items():
 for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)
for nm,a in list(sets.items()):
 if len(nm)==2 and nm[0] in 'BCD':
  for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)

from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import PolynomialFeatures
n=52; q=sets['inv']; T=Y-Y[:n].mean(0); U,S,V=np.linalg.svd(tr[:n],full_matrices=False); Z=np.c_[T[:,0],T[:,1:]@V[:3].T]; sc=Z[:n].std(0); Z=Z/sc
R=[]
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def test(nm,F):
 F=F-F[:n].mean(0); F=F/np.maximum(F[:n].std(0),1e-8); K=F@F[:n].T; E=np.eye(n); 
 for r in [.002,.08,.6]:
  I=cho_solve(cho_factor(K[:n]+r*E),E); a=I@Z[:n]; loo=(a/np.maximum(I.diagonal()[:,None]-1/(n*r),1e-8))*sc; e=(K[n:]@a-Z[n:])*sc
  R.append((np.abs(loo[:,0]).mean(),np.abs(loo[:,1:]@V[:3]).mean(),np.abs(e[:,0]).mean(),np.abs(e[:,1:]@V[:3]).mean(),nm,r))
for tag in ['x','z','inv']:
 q=sets[tag]; P=PolynomialFeatures(3,include_bias=False).fit_transform(q)
 for pn,p in [('lin',P[:,:4]),('quad',P[:,:14]),('cubic',P)]:
  for prefix,g in [('q',q),('Bq',B(q)),('IBq',(q-wb[0])@np.linalg.pinv(wb[1:])),('B0q',q@wb[1:])]:
   cs=[('Cq',C(q)),('Cg',C(g)),('prodq',prod(q)),('prodg',prod(g)),('q',q),('g',g)]; ds=[('Dq',D(q)),('Dg',D(g)),('cosq',cos(q)),('cosg',cos(g)),('sin1q',np.sin(1.25*q)),('trigq',np.c_[cos(q),np.sin(2.4*q)])];
   for (cn,c),(dn,d) in it.product(cs,ds):
    base=np.c_[p,c,d]; cd=cross(c,d); c2=PolynomialFeatures(2,include_bias=False).fit_transform(c)[:,c.shape[1]:];d2=PolynomialFeatures(2,include_bias=False).fit_transform(d)[:,d.shape[1]:];
    for name,F in [('squares',np.c_[c2,d2]),('sqcross',np.c_[c2,d2,cd]),('sqdiag',np.c_[c2,d2,c*d] if c.shape==d.shape else c2),('qf',np.c_[cross(q,c),cross(q,d)]),('mixed',np.c_[cross(q,c),cd]),('sumprod',np.c_[c2,d2,(c*d).sum(1)] if c.shape==d.shape else c2),('rat',np.c_[c/(1+.22*np.sum(d*d,1,keepdims=True)),d/(1+.22*np.sum(c*c,1,keepdims=True))]),('all',np.c_[c2,d2,cd,cross(q,c),cross(q,d)])]:test((tag,pn,prefix,cn,dn,name),np.c_[base,F])
for k in range(6):
 key=(lambda a:a[k]) if k<4 else (lambda a:a[0]+a[1]) if k==4 else (lambda a:a[2]+a[3]); print('best',k,sorted(R,key=key)[:4])
np.set_printoptions(precision=4,suppress=True); print('traj range',np.min(tr,0),np.max(tr,0)); print('temporal components',V[:3]); print('summary',s['train']['summary'],np.mean(Y[:,[0,-1]],0),np.std(Y[:,[0,-1]],0)); print('D spectral',np.linalg.svd(wc2[1:])[1]); print('B spectral',np.linalg.svd(wb[1:])[1]); print('C spectral',np.linalg.svd(wc1[1:])[1])
