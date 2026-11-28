exec(open('/tmp/probe05.py').read().split("c,keep=")[0])
c,keep=a._calibration(s['panel_a']); z=(x-c[0])/np.diag(c[1:5]); b=a._operator_b(s['panel_b'],z)
rng=np.random.default_rng(81)
p=s['panel_c']; u=p['u']; v=p['v1']
def rational(w,par):
 h=w*np.exp(par[:4])+par[4:]; return h/(1+h*h)
def d(w,par):return np.c_[aff(w),rational(w,par)]
def res(par):
 g=d(u,par);return (g@fit(g,v)-v).ravel()
opts=[least_squares(res,p0,bounds=(-3,3),max_nfev=120) for p0 in [np.r_[np.full(4,-.15),np.zeros(4)]]+[rng.normal(0,.35,8) for _ in range(4)]]
par=min(opts,key=lambda r:r.cost).x; D=d(u,par); C=fit(D,v); print('C1 pars',np.round(par,3),'rms',np.std(res(par)),'coef',np.round(C,3));
Q=fit(np.c_[aff(u),prod(u)],p['v2']); print('C2 coef',np.round(Q,3),'rms',np.std(np.c_[aff(u),prod(u)]@Q-p['v2']))
def c1(w):return d(w,par)@C
def c2(w):return np.c_[aff(w),prod(w)]@Q
nodes={'z':z,'b':b,'c':c1(z),'q':c2(z),'bc':c1(b),'bq':c2(b),'cq':c2(c1(z)),'qc':c1(c2(z)),'bcq':c2(c1(b))}
F={}
for k,w in nodes.items():
 F[k]=aff(w); F[k+'2']=np.c_[aff(w),prod(w)]; F[k+'sat']=np.c_[aff(w),np.tanh(w),w/(1+w*w)]
for k in ['b','c','q','bc','bq','cq','qc','bcq']:
 for base in ['z','b','c','q']:
  if base==k:continue
  v,w=nodes[base],nodes[k]; F[base+'+'+k]=np.c_[aff(v),w];F[base+'*'+k]=np.c_[aff(v),w,(v[:,:,None]*w[:,None,:]).reshape(len(z),-1)]
Y=np.c_[y,T]; splits=[(np.arange(n),np.arange(n,72))]+[(v[:56],v[56:]) for v in [rng.permutation(72) for _ in range(5)]]
rows=[]
for name,f in F.items():
 e=[]
 for j,k in splits:
  p=f[k]@fit(f[j],Y[j]); e.append(abs(p-Y[k]).mean(axis=0))
 e=np.array(e); rows.append((name,e[0,0],np.max(e[:,0]),e[0,1:].mean(),e[:,1:].mean(1).max()))
for i in [2,4]:
 print('best by',i);print(*[(r[0],*np.round(r[1:],5)) for r in sorted(rows,key=lambda r:r[i])[:16]],sep='\n')
