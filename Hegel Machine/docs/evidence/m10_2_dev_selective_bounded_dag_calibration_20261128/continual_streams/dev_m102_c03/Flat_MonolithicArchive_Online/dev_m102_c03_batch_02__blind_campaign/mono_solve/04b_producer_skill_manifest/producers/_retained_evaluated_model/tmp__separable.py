exec(open('/tmp/discover.py').read().split('for nm,v')[0]); from scipy.optimize import least_squares
func={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'exp':lambda u:np.exp(-u*u),'rat':lambda u:1/(1+u*u),'soft':lambda u:u/(1+abs(u))}
def phi(z,k,f,linear):
 u=z*f; cols=[np.ones(len(z))]+([z] if linear else [])
 for j in k:cols.append(func[j](u))
 return np.column_stack(cols)
def fit_sep(p,v,k,lin,I):
 def fun(logf,ret=False):
  f=np.exp(logf); F=phi(p,k,f,lin); W=np.linalg.lstsq(F[I],v[I],rcond=1e-10)[0]; return (F@W-v,F,W,f) if ret else (F[I]@W-v[I]).ravel()
 sol=least_squares(fun,np.log([1.8]*4),bounds=(np.log(.12),np.log(6)),max_nfev=150)
 return fun(sol.x,True)
mods=[]
for j in ('v1','v2'):
 v=C[j]; rows=[]
 for k,lin in product(['sin','cos','tanh','atan','exp','rat','soft',('sin','cos'),('tanh','cos'),('tanh','exp'),('sin','exp')],[0,1]):
  if isinstance(k,str): k=(k,)
  rr,F,W,f=fit_sep(p,v,k,lin,np.arange(18)); rows.append((float(rms(rr[18:].ravel())),float(rms(rr[:18].ravel())),k,lin,f))
 print(j,*sorted(rows)[:10],sep='\n')
 mods.append([(k,l,*fit_sep(p,v,k,l,np.arange(24))[2:]) for _,_,k,l,_ in sorted(rows)[:3]])
Z={'z':z,'b':aff(z)@b,'bi':(z-b[0])@np.linalg.inv(b[1:])}; roots=Z.copy()
for nm,u in roots.items():
 for i,models in enumerate(mods):
  for m,(k,l,W,f) in enumerate(models):
   F=phi(u,k,f,l); V=F@W; Z[f'{i}{m}({nm})']=V; Z[f'{i}{m}native({nm})']=F[:,-4:]; Z[f'b{i}{m}({nm})']=aff(V)@b
rows=[]
for nm,u in Z.items():
 for typ in ('poly','joint','warp'):
  for d in (1,2,3):
   if typ=='poly': F=pol(u,d)
   if typ=='joint': F=np.c_[pol(u,d),pol(z,d),np.einsum('ni,nj->nij',u,z).reshape(len(z),16)]
   if typ=='warp': F=np.c_[pol(u,d),pol(np.tanh(u),d),pol(np.sin(u),d)]
   F/=np.maximum(rms(F[:n]),1e-7); K=F[:n]@F[:n].T; Kt=F[n:]@F[:n].T
   for lam in (.0001,.003,.03,.3):
    pred=Kt@np.linalg.solve(K+lam*np.eye(n),Y[:n]); err=np.r_[mae(pred-Y[n:]),mae(((pred-Y[n:])[:,1:]@q.T).ravel())];rows.append((nm,typ,d,lam,err))
for i in (0,3):print('TOP',i,*sorted(rows,key=lambda a:a[-1][i])[:9],sep='\n')
