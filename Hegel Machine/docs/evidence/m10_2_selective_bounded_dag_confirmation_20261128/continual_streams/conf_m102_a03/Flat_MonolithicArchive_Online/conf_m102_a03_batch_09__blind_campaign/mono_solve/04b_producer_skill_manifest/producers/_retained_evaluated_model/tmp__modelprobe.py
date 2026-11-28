exec(open('/tmp/probe.py').read().split('m=get(2);')[0])
from scipy.optimize import least_squares, root
np.set_printoptions(precision=5,suppress=True,linewidth=150)
u=s['panel_a']['u']; v=s['panel_a']['v']; ok=np.linalg.norm(v-u,axis=1)<4*np.median(np.linalg.norm(v-u,axis=1));
def calfeatures(u,p,f,g):
 a=u.ravel(); c=np.roll(u,2,axis=1).ravel(); I=np.tile(np.eye(4),(len(u),1)); return np.c_[I,I*a[:,None],f(p[0]*a),g(p[1]*c)]
best=None
for f in [np.tanh,np.arctan,np.sin]:
 for g in [np.tanh,np.arctan,np.sin]:
  def res(p):
   F=calfeatures(u[ok],p,f,g);return F@fit(F,v[ok].ravel())-v[ok].ravel()
  p=least_squares(res,[2.,2.],bounds=([.15,.15],[6,6])).x; r=np.std(res(p)); print(f.__name__,g.__name__,p,r)
  if best is None or r<best[0]:best=(r,p,f,g)
r,p,f,g=best; C=fit(calfeatures(u[ok],p,f,g),v[ok].ravel());print('cal C',C, 'p',p)
def decode(x):return np.array([root(lambda z: calfeatures(z[None],p,f,g)@C-row, row).x for row in x])
Z=decode(X); print('cal decode residual',np.max(abs((calfeatures(Z,p,f,g)@C).reshape(-1,4)-X)))
b=s['panel_b']; B=fit(aff(b['u']),b['v']); W=aff(Z)@B
op=get(2)._panel_operators(s['panel_c']); T=s['eval_times']; _,_,VR=np.linalg.svd(t['traj'][:52],full_matrices=False); YY=np.c_[t['y'],t['traj']@VR[:2].T]
def score(F):
 R=F@fit(F[:52],YY[:52])-YY;return np.r_[np.abs(R[:52,0]).mean(),np.abs(R[52:,0]).mean(),np.abs(R[:52,1:]@VR[:2]).mean(),np.abs(R[52:,1:]@VR[:2]).mean()]
levels={'x':X,'z':Z,'w':W,'c(z)':op[0](Z),'d(z)':op[1](Z),'c(w)':op[0](W),'d(w)':op[1](W)}
for name,Q in levels.items():
 print(name)
 for d in range(1,4):
  F=np.column_stack([np.ones(len(Q))]+[np.prod(Q[:,ij],axis=1) for j in range(1,d+1) for ij in combinations_with_replacement(range(4),j)]);print(d,score(F))
 for fun in [np.tanh,np.sin,np.arctan]:
  for fr in [.7,1,1.5,2.15,3]:print(fun.__name__,fr,score(np.c_[aff(Q),fun(fr*Q)]))
print('y versus modes linear',score(np.c_[aff(Z),op[0](Z),op[1](Z)]),score(np.c_[aff(Z),op[0](W),op[1](W)]))
