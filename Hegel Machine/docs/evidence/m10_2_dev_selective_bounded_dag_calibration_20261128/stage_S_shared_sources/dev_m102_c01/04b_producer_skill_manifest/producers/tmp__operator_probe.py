exec(open('/tmp/structured.py').read().split('lin=lambda')[0])
p=s['panel_c']; u=p['u']; D=np.c_[np.ones(len(u)),u]; T=s['eval_times']; _,_,V=np.linalg.svd(t[:52],full_matrices=False)
for name in ('v1','v2'):
 v=p[name]; print(name,'linear residual singulars',np.linalg.svd(v-D@lstsq(D,v)[0],compute_uv=False));
 for tag,fn in [('sep2',lambda z:np.c_[z,z**2]),('sep3',lambda z:np.c_[z,z**2,z**3]),('sep4',lambda z:np.c_[z,z**2,z**3,z**4]),('trig',lambda z:np.c_[np.sin(z),1-np.cos(z)]),('rational',lambda z:np.c_[z,z**2]/np.tile(1+.22*z*z,(1,2))),('radial',lambda z:np.c_[z,z**2]/(1+.22*np.sum(z*z,axis=1,keepdims=True))),('tanh',lambda z:np.c_[np.tanh(z),np.tanh(z)**2]),('abs',lambda z:np.c_[z,abs(z)]),('sin3',lambda z:np.c_[z,np.sin(3*z),np.cos(3*z)])]:
  F=np.c_[np.ones(len(u)),fn(u)]; w=lstsq(F,v)[0]; err=v-F@w; h=np.sum(np.linalg.svd(F,full_matrices=False)[0]**2,axis=1); print(tag,np.std(err),np.sqrt(np.mean((err/(1-h[:,None]))**2)))
# Explore whether isotropic kernel predictions can be improved at the domain boundary by a non-decaying polynomial trend.
for j in range(4):
 o=np.argsort(x[:,j]);folds.extend([(o[12:],o[:12]),(o[:-12],o[-12:])]);
o=np.argsort(np.linalg.norm(x,axis=1));folds.append((o[:-14],o[-14:])); z0=[x,a,inv,b(a),b(x),c1(a),c2(a),c2(b(a))]; cases=[]
for iz,z in enumerate(z0):
 d=cdist(z,z,'sqeuclidean')
 for k0 in (1.8,3.,6.,12.,24.):
  for c2w in (.1,.5,2.):
   for c3w in (0.,.05,.2):
    P=P3(x); K=1+x@x.T+c2w*(P[:,4:14]@P[:,4:14].T)+c3w*(P[:,14:]@P[:,14:].T)+3*np.exp(-d/k0)
    for reg in (.0002,.0008,.003,.012):
     e=calc(K,reg); cases.append((e[0,0],e[0,1],e[1:5,0].mean(),e[5:,0].max(),e[5:,1].max(),e[5:,2].max(),iz,k0,c2w,c3w,reg))
for idx in (3,4,5):
 print('ranking',idx); print(*sorted(cases,key=lambda r:r[idx]+.3*r[2]+.2*r[0])[:8],sep='\n')
