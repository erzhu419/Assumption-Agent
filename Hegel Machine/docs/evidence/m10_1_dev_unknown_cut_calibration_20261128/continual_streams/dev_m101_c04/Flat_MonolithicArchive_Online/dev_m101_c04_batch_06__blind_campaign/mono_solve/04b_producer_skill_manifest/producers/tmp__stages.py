import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=56
a_path='/tmp/hegel_continual_archive/dev_m101_c04_batch_02_solver.py'; sp=importlib.util.spec_from_file_location('arc',a_path); arc=importlib.util.module_from_spec(sp); sp.loader.exec_module(arc)
def A(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def rm(x): return np.sqrt(np.mean(x*x))
c=arc._calibrate(s['panel_a']); u=s['panel_a']['u']; v=s['panel_a']['v']; av=np.column_stack([arc._calibration_features(u,j)@c[j] for j in range(4)]); print('A coefficients',c,'sorted residuals',np.round(np.sort(np.linalg.norm(av-v,axis=1)),5)); x=tr['x']; z=arc._unwarp(x,c); print('inverse roundtrip',rm(np.column_stack([arc._calibration_features(z,j)@c[j] for j in range(4)])-x)); b=s['panel_b']; bm=ls(A(b['u']),b['v']); print('B matrix',bm,'singular values',np.linalg.svd(bm[1:],compute_uv=False)); b=A(z)@bm
funcs={
 'tanh':(lambda u,p:np.tanh(p*u),.2,6), 'sin':(lambda u,p:np.sin(p*u),.2,5), 'atan':(lambda u,p:np.arctan(p*u),.2,7), 'sat':(lambda u,p:u/(1+p*u*u),.01,8), 'sqrt_sat':(lambda u,p:u/np.sqrt(1+p*u*u),.01,12), 'oddpower':(lambda u,p:np.sign(u)*abs(u)**p,.2,4), 'evenpower':(lambda u,p:abs(u)**p,.3,5), 'cos':(lambda u,p:np.cos(p*u),.2,6), 'cosh':(lambda u,p:np.cosh(p*u),.2,5), 'exp':(lambda u,p:np.exp(p*u),.01,5), 'softabs':(lambda u,p:np.sqrt(1+p*u*u),.01,8), 'rational_square':(lambda u,p:u*u/(1+p*u*u),.005,8), 'dead':(lambda u,p:np.sign(u)*np.maximum(abs(u)-p,0),0,.8)}
cp=s['panel_c']; u=cp['u']; out={}; opts={}
for key in ['v1','v2']:
 v=cp[key]; cand=[]
 for name,(fun,lo,hi) in funcs.items():
  for extra in [0,1]:
   def F(x,p): return A(np.c_[fun(x,p[0]),x] if extra else fun(x,p[0]))
   def residual(p):
    f=F(u,p); return (f@ls(f,v)-v).ravel()
   fit=min((least_squares(residual,[lo+(hi-lo)*q],bounds=(lo,hi)) for q in [.08,.25,.55,.85]),key=lambda f:f.cost)
   p=fit.x; coef=ls(F(u,p),v); cand.append((rm(fit.fun),name,extra,p[0],coef))
 cand.sort(key=lambda q:q[0]); print('C',key,[tuple(np.round(v,7) if isinstance(v,float) else v for v in t[:4]) for t in cand[:12]])
 opts[key]=cand
 for r,name,extra,p,coef in cand[:5]:
  fun=funcs[name][0]; form=lambda x:A(np.c_[fun(x,p),x] if extra else fun(x,p))@coef
  for label,w in [('z',z),('b',b),('x',x)]:out[key+'_'+label+'_'+name+str(extra)]=form(w)
print('state lead fit and y quadratic/linear')
sets={'raw':x,'z':z,'b':b,**out}
for label,w in sets.items():
 f=A(w); q=np.column_stack([w[:,i]*w[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]); fq=np.c_[f,q]; ta=tr['traj']; y=tr['y']; st=np.mean(abs(f[n:]@ls(f[:n],ta[:n])-ta[n:])); sq=np.mean(abs(fq[n:]@ls(fq[:n],ta[:n])-ta[n:])); yl=np.mean(abs(f[n:]@ls(f[:n],y[:n])-y[n:])); yq=np.mean(abs(fq[n:]@ls(fq[:n],y[:n])-y[n:])); print(label,*np.round([st,sq,yl,yq],5))
np.savez('/tmp/stages.npz',x=x,z=z,b=b,**out)
