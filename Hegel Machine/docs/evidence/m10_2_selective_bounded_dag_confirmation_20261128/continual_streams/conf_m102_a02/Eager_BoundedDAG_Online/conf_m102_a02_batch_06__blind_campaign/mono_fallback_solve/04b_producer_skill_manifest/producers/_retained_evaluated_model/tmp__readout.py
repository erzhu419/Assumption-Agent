exec(open('/tmp/graph.py').read().split('D={}')[0]); z=F['inv']; p=s['panel_b']; u=p['u']; V=p['v']; H=np.c_[y,T]; n=len(y)
f=__import__('scipy').special.erf; e,w,B=panel_fit(p,'v',f,0); g=np.c_[aff(z),f(w*z)]@B; v=z*np.roll(g,1,axis=1)
sp=np.maximum(abs(z).max(1),abs(g).max(1)); splits=[(np.arange(52),np.arange(52,72))]+[(a[:52],a[52:]) for a in [np.random.default_rng(i).permutation(n) for i in range(10)]]+[(np.argsort(sp)[:52],np.argsort(sp)[52:])]
def cv(A,h=y):
 E=np.array([np.mean(abs(A[j]@fit(A[i],h[i])-h[j])) for i,j in splits]); return np.round([E[0], E[1:-1].mean(), E[1:-1].max(),E[-1],np.mean(abs(A@fit(A,h)-h))],5)
D=aff(np.c_[g,v]); D0=aff(np.c_[g,v.sum(1)]); D1=aff(np.c_[g,z,v]); D2=aff(np.c_[g,z,(z[:,:,None]*g[:,None,:]).reshape(n,-1)]); R=H-D@fit(D,H)
print('range',np.quantile(z,[0,.05,.95,1],axis=0),'panel range',u.min(0),u.max(0));print('coefs y T',fit(D,H).T,'baseline',*[cv(a) for a in [D0,D,D1,D2]],'traj',*[cv(a,T) for a in [D0,D,D1,D2]])
for j in np.argsort(abs(R[:,0]))[-12:][::-1]:print('e',j,round(R[j,0],4),round(R[j,-1],4),z[j],g[j], v[j].sum(),y[j])
read=D@fit(D,H); modes=T@np.linalg.svd(T,full_matrices=False)[2][:3].T
G={'g':g,'z':z,'v':v,'zg':np.c_[z,g],'Y':read[:,:1], 'T':read[:,-1:], 'sumv':v.sum(1)[:,None]}
C={}
for k,a in G.items():
 for l,b in {'quad':a*a,'cube':a**3,'four':a**4,'abs':abs(a),'tanh':np.tanh(a),'sin':np.sin(a),'rat':a/(1+a*a),'sqrat':a*a/(1+a*a),'fullquad':poly(a)[:,1+a.shape[1]:]}.items():C[k+' '+l]=b
for k,a in {'g':g,'z':z,'v':v}.items():
 for l,b in {'g':g,'z':z,'v':v}.items():
  for q in range(4): C[f'{k}*{l}roll{q}']=a*np.roll(b,q,axis=1)
for i in range(4):
 for k,a in {'g':g,'z':z}.items():
  for l,b in {'g':g,'z':z}.items():C[f'v*{k}{l}{i}']=v*a*np.roll(b,i,axis=1)
for k,A in [('D0',D0),('D',D),('D1',D1)]:
 print(k,'y',sorted((tuple(cv(np.c_[A,b])),l) for l,b in C.items())[:18]); print(k,'T',sorted((tuple(cv(np.c_[A,b],T)),l) for l,b in C.items())[:10]);print(k,'mean y',sorted((tuple(cv(np.c_[A,b])),l) for l,b in C.items())[:0])
# fit plausible exact panel B bases, including projected rank-one nonlinear correction in each coordinate
J=aff(u); E=V-J@fit(J,V); print('panel residual decomposition',np.linalg.svd(E,compute_uv=False));
for base in ['erf','tanh','rat']:
 f={'erf':__import__('scipy').special.erf,'tanh':np.tanh,'rat':lambda a:a/(1+a*a)}[base]; e,w,B=panel_fit(p,'v',f,0); K=np.c_[J,f(w*u)]; re=V-K@fit(K,V); print(base,'resid sv',np.linalg.svd(re,compute_uv=False));
 for name,add in [('pair',poly(u)[:,5:]),('cross',u*np.roll(u,1,axis=1)),('quint',u**5),('proj-tanh',np.tanh(J@fit(J,V))),('proj-sq',(J@fit(J,V))**2),('proj-cube',(J@fit(J,V))**3)]:
  L=np.c_[K,add]; Q=np.linalg.svd(L,full_matrices=False)[0]; rr=V-L@fit(L,V); print(name,np.mean(rr**2)**.5,np.mean((rr/(1-(Q*Q).sum(1)[:,None]))**2)**.5)
