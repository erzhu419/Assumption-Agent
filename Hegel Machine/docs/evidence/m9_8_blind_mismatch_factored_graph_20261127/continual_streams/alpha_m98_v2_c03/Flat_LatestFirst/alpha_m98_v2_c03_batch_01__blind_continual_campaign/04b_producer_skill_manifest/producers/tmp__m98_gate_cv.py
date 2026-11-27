exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
X=xx; r=X@b[3:]; y=target; n=len(y); lab=ch['lab_id']; R=y-(b[2]+r)
A=np.c_[np.ones(n),X]; u=X[:,0]+.5*X[:,1]; v=X[:,1]; z=X[:,3]
F=[('lab'+str(g),lab==g) for g in np.unique(lab)]+[('edge'+str(j),abs(X[:,j])>=np.sort(abs(X[:,j]))[-2]) for j in range(4)]
for i in np.argsort(r): print('amp',i,np.round(np.r_[X[i],r[i],R[i],R[i]/r[i]],4))
def fit(B,y,k,pen=0.):
 M=B[k]; return np.linalg.lstsq(np.r_[M,np.sqrt(pen)*np.eye(M.shape[1])],np.r_[y[k],np.zeros(M.shape[1])],rcond=1e-9)[0]
lib={}
def add(name,Q,base=A,offset=0.,ridge=0.): lib[name]=(np.column_stack((base,Q)),np.broadcast_to(offset,y.shape),ridge)
empty=np.empty((n,0)); add('linear',empty); add('cubic',np.c_[u**3,v*v*z]); add('rational winner',((X[:,1]+X[:,2])**3/(1+(v-z)**2))[:,None]); add('rational alternate',((X[:,0]+2*v)**3/(1+(v-z)**2))[:,None]);
for c in (0.,.5,1.,1.5):
 for p in (2,3):
  P=np.maximum(abs(X)-c,0)**p; S=P*np.sign(X); W=b[3:];
  add(f'shared signed {c} {p}',S@W); add(f'shared one sided {c} {p}',np.maximum(np.sign(W)*X-c,0)**p@abs(W)); add(f'ratio norm {c} {p}',r*P.sum(1),empty,b[2]+r); add(f'component shrink {c} {p}',S,A,0.,.04)
  add(f'ratio diagonals {c} {p}',r[:,None]*P,empty,b[2]+r,.04)
for c in (.025,.05,.1,.2):
 for p in (2,3):
  for mode in ('norm','weighted norm','max','signed max'):
   S={'norm':np.linalg.norm(X,axis=1),'weighted norm':np.linalg.norm(X*b[3:],axis=1)/np.linalg.norm(b[3:]),'max':abs(X).max(1),'signed max':(np.sign(b[3:])*X).max(1)}[mode]; add(f'{mode} sat {p} {c}', r*S**p/(1+c*S**p),empty,b[2]+r)
for name,Q in [('calibrated hold',hold),('strain0',ch['strain_0']),('mean history',past.mean(1)),('history spread',past.std(1)),('last gap',hold-past[:,-1])]:
 add(name,Q); add(name+' gate',r*Q,empty,b[2]+r)
def err(name,v):
 B,o,p=lib[name]; return B[v]@fit(B,y-o,~v,p)+o[v]-y[v]
def stats(es): return np.round([np.sqrt(np.mean(np.concatenate(es[:4])**2)),np.sqrt(np.mean(np.concatenate(es[4:])**2)),max(np.sqrt(np.mean(e**2)) for e in es)],5)
rank=sorted((tuple(stats([err(name,v) for _,v in F])),name) for name in lib); print('best fixed choices',sorted(rank,key=lambda t:max(t[0][:2]))[:14]); print('baselines',[(name,stats([err(name,v) for _,v in F])) for name in list(lib)[:4]])
for label, names in [('sparse',list(lib)[:4]),('shared',list(lib)[4:])]:
 es=[]; sel=[]
 for _,v in F:
  losses=[]
  for name in names:
   B,o,p=lib[name]; chunks=[]
   for g in np.unique(lab[~v]):
    m=(lab==g)&~v; k=(lab!=g)&~v; chunks.extend(B[m]@fit(B,y-o,k,p)+o[m]-y[m]);
   losses.append(np.sqrt(np.mean(np.square(chunks))))
  name=names[np.argmin(losses)]; sel.append(name); es.append(err(name,v))
 print('nested',label,stats(es),sel)
