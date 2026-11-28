from compact import *
B,C,D,p=operators(s); f=B(x); c=C(f); d=D(f)
# Separate additive terms and pair interactions so their shrinkage need not match.
def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def within(a):
 i,j=np.triu_indices(a.shape[1],1); return a[:,i]*a[:,j]
h=np.sign(f)*(-np.expm1(-1.8*abs(f))); q=f*f/np.sqrt(1+p[2]*f*f)
blocks={'f':f,'c':c,'d':d,'m':cross(f,c),'hh':within(h),'fd':cross(f,d),'cd':cross(c,d),'ff':within(f),'qq':within(q),'hq':cross(h,q),'h3':h**3,'f3':f**3,'qf':cross(q,f),'sq':f*f,'f_sin':np.sin(2*f),'c_sin':np.sin(2*c),'d_sin':np.sin(2*d),'c_tanh':np.tanh(2*c),'h_sin':np.sin(2*h),'c_even':np.cos(2*c),'d_even':np.cos(2*d),'raw_sin':np.sin(2*x),'raw_tanh':np.tanh(2*x)}
basic=['f','c','d','m']
print('panel B',fit(design(s['panel_b']['u']),s['panel_b']['v'])); print('rank 2 temporal',np.linalg.svd(z,full_matrices=False)[2][:2]); print('fit residual baseline', np.std(Y[:52]-design(*(blocks[k] for k in basic))[:52]@fit(design(*(blocks[k] for k in basic))[:52],Y[:52]),axis=0))
def predictor(tr,va,keys,lam,scale):
 A=np.column_stack([blocks[k] * (scale if k not in basic else 1) for k in keys]); g=design(A); H=g[tr]; pen=np.full(g.shape[1],lam); pen[0]=0
 return g[va]@np.linalg.solve(H.T@H+np.diag(pen),H.T@Y[tr])
configs=[]
for add in [(),*((k,) for k in blocks if k not in basic),('hh','ff','hq','qf'),('c_sin','d_sin','c_even','d_even'),('f3','hh','qf'),('c_sin','f_sin','raw_sin')]:
 for lam in (3e-5,.0003,.003):
  for scale in ((1,) if not add else (.3,1,3)):
   configs.append((basic+list(add),lam,scale))
def scores(e): return np.array([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
errors=[]
for cfg in configs:
 ee=np.array([scores(abs(predictor(tr,va,*cfg)-Y[va])) for tr,va in folds]); errors.append(ee)
E=np.array(errors); names=[(','.join(cfg[0][4:]),cfg[1],cfg[2]) for cfg in configs]
for j in (0,1):
 score=E[:,1:6,j].mean(1)+.5*E[:,-1,j]
 for k in np.argsort(score)[:8]: print('rank',j,names[k],E[k,0],E[k,1:6].mean(0),E[k,-1])
# Choose feature set inside each outer training set, including a radial inner slice.
for oi,(tr,va) in enumerate(folds):
 rng=np.random.RandomState(761+oi); subsets=np.array_split(rng.permutation(tr),3); rad=tr[np.argsort(np.linalg.norm(x[tr],axis=1))[-max(10,len(tr)//5):]]; inn=[(np.setdiff1d(tr,j),j) for j in [*subsets,rad]]
 e=np.array([[scores(abs(predictor(t,k,*cfg)-Y[k])) for t,k in inn] for cfg in configs]); objective=e[:,:3].mean(1)+.5*e[:,-1]; ky=np.argmin(objective[:,0]); kz=np.argmin(objective[:,1]); kb=1
 print('nested',oi,names[ky],names[kz],list(E[ky,oi,:1])+list(E[kz,oi,1:]),'base',E[kb,oi])
