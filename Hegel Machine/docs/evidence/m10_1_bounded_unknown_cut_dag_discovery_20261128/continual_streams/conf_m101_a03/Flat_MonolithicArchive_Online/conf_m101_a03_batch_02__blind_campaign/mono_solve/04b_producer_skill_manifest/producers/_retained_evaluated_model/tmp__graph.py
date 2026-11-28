exec(open('/tmp/explore.py').read().split('u,v=(s')[0])
from scipy.optimize import least_squares, root

def calibrate(panel,xx):
 u,v=panel['u'],panel['v']; nxt=np.roll(u,-1,axis=1)
 keep=np.argsort(np.linalg.norm(v-A(u)@fit(A(u),v),axis=1))[:45]
 c=np.array([fit(np.c_[A(u[:,i:i+1]),nxt[:,i],u[:,i]*nxt[:,i],nxt[:,i]**3][keep],v[keep,i]) for i in range(4)]).T
 def forward(z):
  r=np.roll(z,-1,axis=-1); return c[0]+c[1]*z+c[2]*r+c[3]*z*r+c[4]*r**3
 m=np.diag(c[1])+np.roll(np.diag(c[2]),1,axis=0)
 # Construct the affine matrix directly in row-vector convention.
 m=np.diag(c[1]); m[(np.arange(4)+1)%4,np.arange(4)]=c[2]
 z=np.array([root(lambda q:forward(q)-v0,(v0-c[0])@np.linalg.inv(m)).x for v0 in xx]); print('A panel/inversion RMS',np.std((forward(u)-v)[keep]),np.std(forward(z)-xx))
 return z

def cfit(panel):
 u=panel['u']; au=A(u); v=panel['v1']; out=[]
 for family in ['tanh','tanh+aff','sin','sin+aff']:
  cs=[]; es=[]
  for vi in v.T:
   f=np.tanh if family.startswith('tanh') else np.sin
   def design(w,t):
    k=f(A(w)@t); return np.c_[A(w),k] if '+' in family else np.c_[np.ones(len(w)),k]
   def res(t):
    d=design(u,t); return d@fit(d,vi)-vi
   init=fit(au,vi); r=min((least_squares(res,p,max_nfev=150) for p in [init,2*init,4*init,np.r_[0,2*init[1:]]]),key=lambda a:np.linalg.norm(a.fun)); cs.append((r.x,fit(design(u,r.x),vi))); es.append(np.std(r.fun))
  print('C1',family,es, 'params',cs if max(es)<.001 else '')
  out.append((family,cs))
 return out
xx=np.r_[x,s['test']['x']]; z=calibrate(s['panel_a'],xx)
b=s['panel_b']; cb=fit(A(b['u']),b['v']); w=A(z)@cb; ib=(z-cb[0])@np.linalg.inv(cb[1:]); pc=s['panel_c']; c1=cfit(pc); c2=fit(poly(pc['u'],2),pc['v2']); print('C2 terms',c2)
for name,q0 in [('z',z),('w',w),('ib',ib),('x',xx)]:
 q=q0[:len(x)]; a=A(q); f2=poly(q,2)
 print(name,'ranges',q0.min(0),q0.max(0),'poly scores',score(f2),score(poly(q,3)))
 print('quadratic coefficients',fit(f2[:52],Y[:52])[:,[0,1,-1]].T)
 for fam,cs in c1[:2]:
  h=np.column_stack([(np.c_[a,np.tanh(a@t)] if '+' in fam else np.c_[np.ones(len(q)),np.tanh(a@t)])@beta for t,beta in cs]); g=f2@c2
  for nm,f in [('h',A(h)),('h2',poly(h,2)),('q+h',np.c_[a,h]),('q+h+g',np.c_[a,h,g]),('q2+h',np.c_[f2,h]),('h+g',np.c_[A(h),g]),('q+hg',np.c_[a,h,g,h*g]),('q+tanh',np.c_[a,np.tanh(q)])]: print(nm,fam,score(f))
