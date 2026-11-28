exec(open('/tmp/structure.py').read().split('for rates')[0])
from itertools import combinations
F=-np.expm1(-t[:,None]*[.22,1.25]); q=ls(F,z.T).T
print('time noise',np.sqrt(np.mean((q@F.T-z)**2)),np.linalg.svd(q,compute_uv=False)); print('amplitude correlations',np.corrcoef(np.c_[q,y].T))
for r in [1,2]:
 def res(p):
  f=-np.expm1(-t[:,None]*np.exp(p));return (f@ls(f,z[:52].T)-z[:52].T).ravel()
 o=least_squares(res,np.log(np.geomspace(.22,1.25,r)));print('rate fit',np.exp(o.x),np.std(o.fun))
B=np.c_[y,q]; pairs=list(combinations(range(4),2)); triples=list(combinations(range(4),3)); folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=27).split(x))
def prod(w,ix):return np.column_stack([np.prod(w[:,j],axis=1) for j in ix]);
def ev(g):
 out=[]
 for i,j in folds:
  pred=g[j]@ls(g[i],B[i]);out.append([np.mean(abs(pred[:,0]-y[j])),np.mean(abs(pred[:,1:]@F.T-z[j]))]);
 return np.array(out)
base=[]
for family in ['ratio2','ratio3','root2','root3','tanh1','tanh3','sin1','sin3','bare']:
 for c in [.11,.22,.35,.55,.8,1.25]:
  k=3 if family.endswith('3') else 2
  h={'ratio':lambda:u**3/(1+c*abs(u)**k),'root':lambda:u**3/np.sqrt(1+c*abs(u)**k),'tanh':lambda:np.tanh(c*u**(3 if k==3 else 1)),'sin':lambda:np.sin(c*u**(3 if k==3 else 1)),'bar':lambda:u**3}[family.rstrip('123') if family!='bare' else 'bar']()
  for extra in range(7):
   g=np.c_[np.ones(len(u)),h]
   if extra in [1,2,3,4,5,6]:g=np.c_[g,u]
   if extra in [2,3,5,6]:g=np.c_[g,prod(u,pairs)]
   if extra in [3,6]:g=np.c_[g,prod(u,triples)]
   if extra in [4,5,6]:g=np.c_[g,u*u]
   E=ev(g); base.append((E[:,0].mean(),E[:,1].mean(),family,c,extra,E.max(0),E[0]))
for col in [0,1]:
 print('SORT',col)
 for a in sorted(base,key=lambda row: row[col])[:12]: print(a)
for label,g in [('cubic',poly(u,3)),('separable',np.c_[aff(u),u*u,u**3]),('cube ratio',np.c_[aff(u),u**3/(1+.22*u*u)]),('tanh cube',np.c_[aff(u),np.tanh(1.25*u**3)])]:
 print(label,ev(g).round(5)); print('ols',ls(g[:52],B[:52]).round(4))
