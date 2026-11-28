import hegel_probe as h
from hegel_probe import np,p,a,least_squares
h.vi=np.arange(52,72); s=h.study(np.arange(52),h.vi); x=np.r_[s['train']['x'],s['test']['x']]; n=52
lin=a.lin
u,v=p['panel_a'].values(); F=a.one(u); ok=np.ones(len(u),bool)
for _ in range(8):
 A=lin(F[ok],v[ok]); r=np.linalg.norm(v-F@A,axis=1);ok=r<max(4*np.median(r),.001)
print('A',A,'residuals',np.sort(r)); print('A SVD',np.linalg.svd(A[1:],compute_uv=False))
ub,vb=p['panel_b'].values(); B=lin(a.one(ub),vb);print('B svd',np.linalg.svd(B[1:],compute_uv=False))
u=p['panel_c']['u']; v=p['panel_c']['v1']
models={'soft':lambda w,q:w/(1+q*abs(w)), 'tanh':lambda w,q:np.tanh(q*w),'atan':lambda w,q:np.arctan(q*w),'sin':lambda w,q:np.sin(q*w),'asinh':lambda w,q:np.arcsinh(q*w),'root':lambda w,q:w/np.sqrt(1+q*w*w),'rat2':lambda w,q:w/(1+q*w*w),'power':lambda w,q:np.sign(w)*abs(w)**q,'exp':lambda w,q:np.sign(w)*(-np.expm1(-q*abs(w))),'erf':lambda w,q:__import__('scipy').special.erf(q*w)}
fits=[]
for name,fn in models.items():
 for mode in ('shared','separate','full','shared-even'):
  def d(w,q,j):
   z=fn(w,q if mode=='shared' else q[j] if mode=='separate' else q); b=a.one(w) if mode.startswith('shared') or mode=='full' else a.one(w[:,j]); z=z if mode=='full' else z[:,j];return np.c_[b,z,w*w] if mode=='shared-even' else np.c_[b,z]
  def res(q):return np.concatenate([d(u,q,j)@lin(d(u,q,j),v[:,j])-v[:,j] for j in range(4)])
  qq=min((least_squares(res,np.full(4 if mode=='separate' else 1,q),bounds=(.02,10),gtol=1e-12) for q in [.4,1.4,2.7]),key=lambda f:f.cost); err=np.sqrt(np.mean(qq.fun**2));fits.append((err,name,mode,qq.x));print(name,mode,qq.x,err)
Ai=lambda z:(z-A[0])@np.linalg.inv(A[1:]); Ad=lambda z:a.one(z)@A
Bi=lambda z:(z-B[0])@np.linalg.inv(B[1:]); Bd=lambda z:a.one(z)@B
for name,z in {'Ai':Ai(x),'Ad':Ad(x),'AiBi':Bi(Ai(x)),'AiBd':Bd(Ai(x)),'BiAi':Ai(Bi(x)),'BdAi':Ai(Bd(x)),'Bd':Bd(x),'Bi':Bi(x)}.items():
 for label,w in [('raw',z),('sq',np.c_[z,z*z]),('archive',a._panel_maps(p['panel_c'],z)),('arch+z',np.c_[z,a._panel_maps(p['panel_c'],z)])]:
  for std in [True,False]:
   w1=(w-w[:n].mean(0))/np.maximum(w[:n].std(0),.1) if std else w
   print(name,label,std,h.score(w1,s))
