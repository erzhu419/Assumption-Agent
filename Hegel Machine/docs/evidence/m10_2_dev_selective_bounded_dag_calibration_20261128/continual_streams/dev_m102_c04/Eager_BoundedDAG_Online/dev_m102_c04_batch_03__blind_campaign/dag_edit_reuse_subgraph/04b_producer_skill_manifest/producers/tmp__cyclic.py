exec(open('/tmp/modelprobe.py').read().split('o,M=')[0]); from scipy.optimize import root
u,v=A['u'],A['v']; keep=np.ones(len(u),bool)
for t in range(5):
 w=ls(a(u)[keep],v[keep]); d=np.linalg.norm(a(u)@w-v,axis=1); keep=d<4*np.median(d)
pop,rop=m['_panel_operators'](C)
def terms(z,j,degree):
 b,c=z[:,j],z[:,(j+1)%4]; return np.column_stack([np.ones(len(z))]+[b**i*c**(k-i) for k in range(1,degree+1) for i in range(k+1)])
def check(degree):
 ws=[ls(terms(u,j,degree)[keep],v[keep,j]) for j in range(4)]
 def forward(z):return np.column_stack([terms(z,j,degree)@ws[j] for j in range(4)])
 e=forward(u)-v; print('degree',degree,'fit',np.sqrt(np.mean(e[keep]**2)),'largest',np.max(abs(e[keep])), 'weights',np.array(ws).T if degree<=4 else '')
 q0=np.linalg.solve(w[1:].T,(x-w[0]).T).T
 q=np.array([least_squares(lambda z:forward(z[None])[0]-xx, np.clip(zz,-1.49,1.49),bounds=(-1.5,1.5),max_nfev=40).x for xx,zz in zip(x,q0)])
 print('inversion',np.max(abs(forward(q)-x)),np.quantile(q,[0,.01,.99,1]))
 def score(c,d):
  yy=ridge(c[:n],y[:n],c[n:],1e-6); out=frozen({'node_00':dict(common=c,offdiag=d,y_hat=yy),'node_01':temporal}); return np.mean(abs(yy-y[n:])),np.mean(abs(out['state_pred']-traj[n:]))
 pairs=[]; b=ls(a(B['u']),B['v']); bm=b[1:]; bo=b[0]
 coords={'x':x,'q':q,'qB':q@bm+bo,'qBinv':np.linalg.solve(bm.T,(q-bo).T).T,'xB':x@bm+bo,'xBinv':np.linalg.solve(bm.T,(x-bo).T).T}
 for pn,z in coords.items():
  for rn,t in coords.items():
   p,r=pop(z),rop(t); diag=p*r; c,d=m['_features'](p,r)
   for name,common in [('archive',c),('diag',np.c_[p,r,diag]),('all',np.c_[p,r,diag,d])]:
    ma,ta=score(common,d); pairs.append((max(ma,ta),ma,ta,pn,rn,name))
 print('top',sorted(pairs)[:5])
for deg in [2,3,4,5]:check(deg)
