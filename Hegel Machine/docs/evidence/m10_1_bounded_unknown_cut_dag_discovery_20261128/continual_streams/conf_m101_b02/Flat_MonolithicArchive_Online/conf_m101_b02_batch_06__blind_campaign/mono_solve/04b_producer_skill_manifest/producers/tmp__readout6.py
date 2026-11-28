from probe6 import *
from scipy.optimize import least_squares

def nodes(x):
 p=s['panel_a']; _,coef=calfit('cross'); u,v=p['u'],p['v']; g=calbasis(u,'cross'); keep=np.linalg.norm(g@coef-v,axis=1)<.002
 for j in range(4):
  k=abs(coef[:,j])>.001; coef[~k,j]=0; coef[k,j]=fit(g[keep][:,k],v[keep,j])
 init=np.linalg.solve(coef[1:5].T,(x-coef[0]).T).T
 a=np.array([least_squares(lambda z:calbasis(z[None,:],'cross')[0]@coef-row,z).x for z,row in zip(init,x)])
 def op(p,k,f,x,q):
  p=s[p]; fun=fs[f]; return fun(x,q)@fit(fun(p['u'],q),p[k])
 b=op('panel_b','v','shrink',a,.35)
 return a,b,op('panel_c','v1','archpair',b,0),op('panel_c','v2','evenrat',b,.39)

def design(ns, recipe):
 g=[]; names=[]
 def add(z,n):
  g.append(np.asarray(z).reshape(len(ns['a']),-1)); names.extend(n)
 for term in recipe.split():
  if len(term)==1: add(ns[term],[term+str(i) for i in range(4)]); continue
  if term[0]=='@': add((ns[term[1]]*ns[term[2]]).sum(1),[term]); continue
  a,b=term; z,w=ns[a],ns[b]; ij=list(combinations_with_replacement(range(4),2)) if a==b else [(i,j) for i in range(4) for j in range(4)]
  add(np.column_stack([z[:,i]*w[:,j] for i,j in ij]),[f'{a}{i}{b}{j}' for i,j in ij])
 return D(*g),['1']+names

def run():
 x=s['train']['x']; ns=dict(zip('abcd',nodes(x))); ns['x']=x
 Y=np.c_[s['train']['y'],s['train']['traj']]; folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; folds += [(np.r_[0:i,i+18:72],np.arange(i,i+18)) for i in (0,18,36,54)]
 recipes=['a','a b','a c','a d','a c @ac','a c @ac @bc @ad @cd','a c d @ac @ad @cd','a b c d @ac @bc @ad @cd @ab @bd','a b c d','a b c d aa','a b c d ac','a b c d ad','a b c d bc','a b c d cd','a c d ac','a c d ad','a c d ac ad','a b c d ac ad cd','x xx']
 def score(g,lam):
  errs=[]
  for I,J in folds:
   if lam:
    sc=np.maximum(g[I].std(0),1e-9); sc[0]=1; h=g/sc; R=np.eye(g.shape[1])*lam; R[0,0]=0; co=np.linalg.solve(h[I].T@h[I]+R,h[I].T@Y[I]); p=h[J]@co
   else: p=g[J]@fit(g[I],Y[I])
   e=abs(p-Y[J]); errs.append([e[:,0].mean(), e[:,1:].mean()])
  e=np.array(errs); return np.r_[e[0],e[1],e.max(0)]
 for r in recipes:
  g,names=design(ns,r); out=[score(g,l) for l in (0,.001,.03,.3)]; print(r, np.round(out,5).tolist())
  if r in ('a c @ac','a b c d @ac @bc @ad @cd @ab @bd'):
   co=fit(g[:52],Y[:52]); print('coefs',list(zip(names,co[:,[0,1,-1]].round(4).tolist())))
 for other in ('a','x'):
  a=ns[other]; b=D(a,pairs(a),a*a,a**3)
  print('poly',other,[score(b,l).round(5).tolist() for l in (0,.001,.03,.3)])
 t=s['eval_times']; T=Y[:52,1:].T
 def res(q):
  g=-np.expm1(-t[:,None]*q); return (g@fit(g,T)-T).ravel()
 q=least_squares(res,[.4,2],bounds=([.01,.01],[8,8]),gtol=1e-11).x
 print('rates',q,'rms',np.mean(res(q)**2)**.5,'singular',np.linalg.svd(T,compute_uv=False))
 print('node ranges',[(k,v.min(0).round(3),v.max(0).round(3)) for k,v in ns.items()])
if __name__=='__main__': run()
