import importlib.util,pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x,y,tr=(t[k] for k in ('x','y','traj')); N=len(x); ids=np.arange(N)
a,b=m._panel(s['panel_a'],3),m._panel(s['panel_b']); w=m._affine(m._affine(x)@a)@b
rng=np.random.default_rng(228); splits=[('public',ids[:52],ids[52:])]+[(f'random{i}',p[:52],p[52:]) for i,p in enumerate(rng.permutation(N) for _ in range(15))];
p=np.argsort(np.linalg.norm((x-x.mean(0))/x.std(0),axis=1));splits.append(('radial',p[:56],p[56:]))
for j in range(4):
 p=np.argsort(x[:,j]);splits.extend([(f'low{j}',p[12:],p[:12]),(f'high{j}',p[:-12],p[-12:])])
for i,p in enumerate(np.random.default_rng(441).permutation(N) for _ in range(14)):
 splits.append((f'fresh{i}',p[:56],p[56:]))
R={}; E={};
for label,a,b in splits:
 z=(w-w[a].mean(0))/w[a].std(0); vv=(x-x[a].mean(0))/x[a].std(0); D=cdist(z,z,'sqeuclidean'); DX=cdist(x,x,'sqeuclidean'); Q=(1+z@z.T/4)**2; ab=np.ix_(a,b); aa=np.ix_(a,a); ba=np.ix_(b,a); out={};
 out['base']=.3*m._scalar_kernel(x[a],x[b],z[a],z[b],y[a])+.45*m._rational(z[a],z[b],y[a])+.25*m._rational(vv[a],vv[b],y[a]); out['kernel']=m._scalar_kernel(x[a],x[b],z[a],z[b],y[a]);
 for kind in ['m32','m52','gauss','rq']:
  for g in [.025,.06,.13,.27]:
   H=g*D; U=np.sqrt(5*H if kind=='m52' else 3*H); K={'m52':(1+U+U**2/3)*np.exp(-U),'m32':(1+U)*np.exp(-U),'gauss':np.exp(-H),'rq':1/(1+H)}[kind]
   for poly in [0,.1,.6]:
    M=K+poly*Q+.4*np.exp(-.23*DX)
    for noise in [.00025,.003,.015]:out[(kind,g,poly,noise)]=m._universal(M[aa],M[ba],z[a],z[b],y[a],noise)
 for name,Z in [('raw',x),('AB',w),('AB0',w-m._affine(m._affine(np.zeros((1,4)))@m._panel(s['panel_a'],3))@m._panel(s['panel_b']))]:
  p=m._quadratic(Z); r=np.sum(Z**2,axis=1); reg=np.diag(np.sqrt(np.r_[np.zeros(5),np.full(10,.001)]));
  for lin in [0,1]:
   for g in [.1,.25,.45,.7,1.]:
    F=p/(1+g*r[:,None]); F=np.c_[F,Z] if lin else F; reg2=np.diag(np.r_[reg.diagonal(),np.full(4,.04)]) if lin else reg
    coef=np.linalg.lstsq(np.r_[F[a],reg2],np.r_[y[a],np.zeros(len(reg2))],rcond=None)[0]; out[(name,lin,g)]=F[b]@coef
 for key,pred in out.items():R.setdefault(key,[]).append(pred-y[b]);E.setdefault(key,[]).append(abs(pred-y[b]).mean())
def stats(e):
 e=np.asarray(e);return np.round([e[0],e[1:16].mean(),e[:16].max(),e[16:25].max(),e[25:].mean(),e[25:].max()],5)
for key in ['base','kernel']: print(key,stats(E[key]))
for ii in [1,2,3]:
 print('rank',ii)
 score=lambda k:stats(E[k])[ii]+(.15 if ii!=1 else .4)*stats(E[k])[1]
 for key in sorted(E,key=score)[:9]:print(key,stats(E[key]))
for kind in ['raw','AB','AB0']:
 for key in sorted([k for k in E if isinstance(k,tuple) and k[0]==kind],key=lambda k:stats(E[k])[1])[:2]:print(key,stats(E[key]))
np.save('/tmp/challenge_res.npy',R,allow_pickle=True)
