exec(open('/tmp/explore.py').read().split('u,v=(s')[0])
from scipy.optimize import least_squares, root

def calibrate(panel,xx):
 u,v=panel['u'],panel['v']; nxt=np.roll(u,-1,axis=1)
 keep=np.argsort(np.linalg.norm(v-A(u)@fit(A(u),v),axis=1))[:45]
 for _ in range(3):
  c=np.array([fit(np.c_[A(u[:,i:i+1]),nxt[:,i],u[:,i]*nxt[:,i],nxt[:,i]**3][keep],v[keep,i]) for i in range(4)]).T
  keep=np.argsort(np.linalg.norm(v-(c[0]+c[1]*u+c[2]*nxt+c[3]*u*nxt+c[4]*nxt**3),axis=1))[:45]
 def forward(z):
  r=np.roll(z,-1,axis=-1); return c[0]+c[1]*z+c[2]*r+c[3]*z*r+c[4]*r**3
 m=np.diag(c[1])+np.roll(np.diag(c[2]),1,axis=0)
 # Construct the affine matrix directly in row-vector convention.
 m=np.diag(c[1]); m[(np.arange(4)+1)%4,np.arange(4)]=c[2]
 z=np.array([root(lambda q:forward(q)-v0,(v0-c[0])@np.linalg.inv(m)).x for v0 in xx]); print('A panel/inversion RMS',np.std((forward(u)-v)[keep]),np.std(forward(z)-xx))
 return z

xx=np.r_[x,s['test']['x']]; z=calibrate(s['panel_a'],xx); b=s['panel_b']; cb=fit(A(b['u']),b['v']); w=A(z)@cb
p=s['panel_c']; u=p['u']; print('B',cb,'RMS',np.std(b['v']-A(b['u'])@cb)); print('times',s['eval_times'])
for name,f in [('tanh(u),squares',np.c_[A(np.tanh(u)),u*u]),('aff,tanh(u),squares',np.c_[A(u),np.tanh(u),u*u]),('tanh+quad',np.c_[poly(u,2),np.tanh(u)]),('tanh shifted',np.c_[A(np.tanh(u+.3)),u*u]),('saturate',A(u/np.sqrt(1+u*u)))]: print('C1',name,np.std(p['v1']-f@fit(f,p['v1'])))
for name,q in [('z',z[:72]),('w',w[:72]),('invb',((z-cb[0])@np.linalg.inv(cb[1:]))[:72])]:
 print(name)
 f=poly(q,3); print('cubic',score(f)); print('cubic weights',fit(f[:52],Y[:52])[:,[0,1,-1]].T)
 for k,d in [('a+tanh',np.c_[A(q),np.tanh(q)]),('a+sin',np.c_[A(q),np.sin(q)]),('a+cube',np.c_[A(q),q**3]),('quad+cube',np.c_[poly(q,2),q**3]),('quad+tanh',np.c_[poly(q,2),np.tanh(q)]),('quad+sin',np.c_[poly(q,2),np.sin(q)])]:
  print(k,score(d))
 print('fit modes',np.linalg.svd(Y[:,1:],compute_uv=False))
np.savez('/tmp/coords.npz',z=z,w=w,cb=cb)
