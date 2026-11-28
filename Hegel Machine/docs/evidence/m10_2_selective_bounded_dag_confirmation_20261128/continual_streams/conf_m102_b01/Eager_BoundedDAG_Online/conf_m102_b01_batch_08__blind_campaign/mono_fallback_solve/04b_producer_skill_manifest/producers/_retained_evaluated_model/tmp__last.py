import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=np.load('/tmp/operators.npz'); X,Z,Y=d['X'],d['Z'],d['Y']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda a:np.c_[np.ones(len(a)),a]; u,v=[s['panel_b'][j] for j in ('u','v')]; roll=lambda a:np.roll(a,-1,axis=1); Q=np.load('/tmp/newroutes.npz'); sp=importlib.util.spec_from_file_location('p','/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def odd(a,b):return np.sign(a)*abs(a)**b
R={k:w for k,w in np.load('/tmp/routes.npz').items()}; R.update(dict(Q)); R.update(X=X,Z=Z,C=d['C1'],Q=d['C22'])
for label,fn in [('tanh3',lambda x,g:np.tanh(g*x**3)),('asinh3',lambda x,g:np.arcsinh(g*x**3)),('atan3',lambda x,g:np.arctan(g*x**3)),('rat3',lambda x,g:x**3/(1+g*x*x)),('rat6',lambda x,g:x**3/(1+g*x**6)),('pow',lambda x,g:odd(x,g)),('sinh3',lambda x,g:np.sinh(g*x**3))]:
 for aug in ['0','u','r','rr','ucyc','rcyc','rrcyc']:
  def f(x,g):
   a=fn(x,g); b=odd(x,1/3); cols={'0':(), 'u':(x,), 'r':(b,), 'rr':(b,x**3), 'ucyc':(x,x*roll(x)), 'rcyc':(b,x*roll(x)), 'rrcyc':(b,x**3,x*roll(x))}[aug];return np.c_[aff(a),*cols]
  res=lambda g:(f(u,g)@ls(f(u,g),v)-v).ravel(); opt=least_squares(res,[1.25],bounds=(.005,5)); g=opt.x[0]; a=f(u,g); q=ls(a,v); h=(a*np.linalg.pinv(a).T).sum(1); print(label,aug,round(g,5),round(np.std(opt.fun),6),round(np.mean(abs(opt.fun.reshape(-1,4)/(1-h)[:,None])),6)) if np.std(opt.fun)<.005 else None
  for k,x in [('X',X),('Z',Z)]: R[label+aug+k]=f(x,g)@q
 A=aff(fn(u,1.25)); mat=ls(A,v); inv=np.linalg.pinv(mat[1:]);
 for k,x in [('X',X),('Z',Z)]:R[label+'inv'+k]=(x-mat[0])@inv
print('prior standalone',np.mean(abs(m._predict(R['prior'][:72],Y[:52,0],1,'atan')-Y[52:,0])),np.mean(abs(m._predict(R['prior'][:72],Y[:52,1:],0,'inv3')-Y[52:,1:])))
# Search output links and directional rational terms; score every model on a second blocked split as well.
S=[]; rng=np.random.default_rng(623); splits=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))];
for k,w in R.items():
 if np.max(abs(w))>30:continue
 w=w/np.maximum(np.median(abs(w[:72]),axis=0),.06)
 for g in [.3,.7,1.25,2.5,5.0]:
  for name,t in [('asinh',np.arcsinh(g*w)),('root',np.sinh(np.arcsinh(g*w)/3)),('atan',np.arctan(g*w)),('tanh',np.tanh(g*w)),('rational',w/(1+g*w*w)),('cubicroot',odd(w,1/3)),('sinc',np.sin(g*w))]:
   if name=='cubicroot' and g!=1.25:continue
   for typ in range(5):
    base=aff(t) if typ<=1 else np.c_[aff(w),t];
    if typ in (1,3):base=np.c_[base,*[t[:,i]*t[:,j] for i in range(4) for j in range(i,4)]]
    if typ==4:base=np.c_[base,t**3,t*np.roll(t,1,1)]
    for reg in (.0003,.02,.3):
     es=[]
     for tr,te in splits:
      F=base/np.maximum(np.linalg.norm(base[tr],axis=0),1e-9); a=F[tr];p=F[te]@np.linalg.solve(a.T@a+reg*np.eye(a.shape[1]),a.T@Y[tr]); e=np.mean(abs(p-Y[te]),axis=0);es.append((e[0],e[1:].mean()));
     S.append((es,(k,g,name,typ,reg)))
for j in (0,1):
 for st in (0,1):
  print('TOP',j,st, sorted(S,key=lambda e:e[0][0][j] if st==0 else max(e[0][0][j],e[0][1][j]))[:7])
np.savez('/tmp/allroutes.npz',**R)
