from probe import *
from itertools import combinations_with_replacement as cr, combinations
Z=np.load('/tmp/z.npy'); u=s['panel_b']['u']; v=s['panel_b']['v']; uc=s['panel_c']['u'];
cpairs=list(zip(range(4),[1,2,3,0])); cd=lambda x:ic(np.column_stack([x[:,i]*x[:,j] for i,j in cpairs])); cq=ls(cd(uc),s['panel_c']['v1']); C=lambda x:cd(x)@cq
G=opfit(uc,s['panel_c']['v2'],fs['exp'],True,False)[0];
print('C coefficients',cq,'C2')
operators={}
for name in ['sin','erf','tanh']:
 f,e,q=opfit(u,v,fs[name],True,False); operators[name]=f
 d=ic(np.c_[u,fs[name](u*q)]); print(name,e,ls(d,v).round(3))
for nam in ['sin','tanh','rat','erf','asinh']:
 fn=fs[nam]; best=[]
 for j in range(4):
  def d(a,q):return ic(np.c_[a,fn(a@q[:4]+q[4])]);
  def residual(q):
   aa=d(u,q);return aa@ls(aa,v[:,j])-v[:,j]
  bestfit=min((least_squares(residual,q,max_nfev=180) for q in np.vstack([np.c_[2*np.eye(4),np.zeros(4)],np.random.default_rng(1).normal(size=(10,5))])),key=lambda f:f.cost); best.append(bestfit.x)
 def build(nam,best):
  fn=fs[nam]; return lambda a:np.column_stack([ic(np.c_[a,fn(a@q[:4]+q[4])])@ls(ic(np.c_[u,fn(u@q[:4]+q[4])]),v[:,j]) for j,q in enumerate(best)])
 op=build(nam,best); print('ridge',nam,np.sqrt(np.mean((op(u)-v)**2)),np.array(best).round(2)); operators[nam+'-ridge']=op
YY=np.c_[Y,T]; out=[]
for nam,B in operators.items():
 D={'z':Z,'b':B(Z),'c':C(Z),'g':G(Z),'cb':C(B(Z)),'gb':G(B(Z)),'bc':B(C(Z)),'bg':B(G(Z)),'gc':G(C(Z)),'cg':C(G(Z))};
 def score(n,f):
  d=ic(f); c=ls(d[:52],YY[:52]); E=abs(d[52:]@c-YY[52:]).mean(0); ee=abs(d[:52]@c-YY[:52]).mean(0); out.append([*E[[0]],E[1:].mean(),ee[0],ee[1:].mean(),nam,n,d.shape[1]])
 for l in range(1,4):
  for names in combinations(D,l):
   mats=[D[k] for k in names]; base=np.column_stack(mats); score('+'.join(names),base)
   for i,j in cr(range(l),2):
    p=mats[i]*mats[j]; ab=names[i]+'*'+names[j]; score('+'.join(names)+'+dot'+ab,np.c_[base,p.sum(1)]);score('+'.join(names)+'+had'+ab,np.c_[base,p]);score('+'.join(names)+'+cross'+ab,np.c_[base,(mats[i][:,:,None]*mats[j][:,None,:]).reshape(len(Z),-1)])
for key in [lambda a:a[0],lambda a:a[1],lambda a:max(a[:2])]:
 print('best');
 for a in sorted(out,key=key)[:12]:print(np.round(a[:4],5),a[4:])
