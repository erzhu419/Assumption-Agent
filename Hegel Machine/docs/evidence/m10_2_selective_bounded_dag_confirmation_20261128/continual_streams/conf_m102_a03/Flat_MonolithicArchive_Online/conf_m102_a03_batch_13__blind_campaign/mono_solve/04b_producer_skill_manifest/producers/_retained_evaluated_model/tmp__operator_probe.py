import numpy as np,pickle,glob,importlib.util
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('m',glob.glob('/tmp/hegel_continual_archive/*06_solver.py')[0]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];t=s['eval_times']; fit=m._fit; aff=m._affine
cal=m._calibration(s['panel_a']);z=m._decode(tr['x'],cal);print('cal',cal,'z range',z.min(),z.max());u,v=s['panel_a'].values(); b=np.roll(u,-1,axis=1); pred=cal[:4]+u*cal[4:8]+cal[8]*b+cal[9]*u*b+cal[10]*u**3;print('cal residual norms',np.sort(np.linalg.norm(pred-v,axis=1)))
Y=np.c_[tr['y'],tr['traj']]; idx=np.arange(52); val=np.arange(52,72)
def report(n,x):
 a=aff(x);p=a[val]@fit(a[idx],Y[idx]);e=np.abs(p-Y[val]);print(n,np.round([e[:,0].mean(),e[:,1:].mean()],5))
report('z',z)
warps={'lin':(lambda x,p:x,[]),'cubic':(lambda x,p:x+p[0]*x**3,[.2]),'sin':(lambda x,p:np.sin(p[0]*x),[2.15]),'cos':(lambda x,p:np.cos(p[0]*x),[2.15]),'tanh':(lambda x,p:np.tanh(p[0]*x),[1.25]),'atan':(lambda x,p:np.arctan(p[0]*x),[1.25]),'dead':(m._warp,[.22,1.25]),'atan+sin':(lambda x,p:np.c_[x,np.arctan(p[0]*x),np.sin(p[1]*x)],[4.4,2.58]),'x+cos':(lambda x,p:x+p[1]*np.cos(p[0]*x),[1.28,-.65]),'x+sin':(lambda x,p:x+p[1]*np.sin(p[0]*x),[2.15,.22]),'softclip':(lambda x,p:x/(1+abs(p[0])*x*x),[.22]),'oddpow':(lambda x,p:np.sign(x)*np.abs(x)**abs(p[0]),[1.25]),'poly2':(lambda x,p:np.c_[x,x*x,*[x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)]],[])}
for k in ['panel_b','panel_c']:
 u=s[k]['u']
 for vkey,v in s[k].items():
  if vkey=='u': continue
  print(k,vkey); choices=[]
  for n,(f,p) in warps.items():
   def des(x,p):return aff(f(x,p))
   def res(p):a=des(u,p);return (a@fit(a,v)-v).ravel()
   if p:
    try: p=least_squares(res,p,max_nfev=150,bounds=([0,.2],[1,5]) if n=='dead' else (-np.inf,np.inf)).x
    except Exception: continue
   a=des(u,p); coef=fit(a,v); err=np.sqrt(np.mean(res(p)**2)); choices.append((err,n,p,des(z,p)@coef))
  for e,n,p,w in sorted(choices,key=lambda q:q[0])[:6]:
   print(' panel rmse',n,e,p);report('   downstream',w)
print('normalized temporal means',fit(tr['traj'][:,[-1]],tr['traj']).ravel());print('y relative to last',fit(aff(tr['traj'][:,[-1]]),tr['y']), np.std(tr['y']-aff(tr['traj'][:,[-1]])@fit(aff(tr['traj'][:,[-1]]),tr['y'])))
