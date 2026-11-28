import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def design(*xs): return np.column_stack([np.ones(len(xs[0])),*xs])
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def fit(u,v,f,robust=False):
 a=f(u); w=np.ones(len(u));
 for _ in range(12 if robust else 1):
  b=ls(a*w[:,None],v*w[:,None]); d=np.linalg.norm(a@b-v,axis=1); w=np.sqrt(np.minimum(1,2.5*np.median(d)/np.maximum(d,1e-12)))
 return lambda x:f(x)@b
p=s['panel_a']; print('A matrix',ls(design(p['u']),p['v'])); print('A residual',np.round(np.sort(np.linalg.norm(design(p['u'])@ls(design(p['u']),p['v'])-p['v'],axis=1)),5))
A=fit(p['u'],p['v'],design,True); Ai=fit(p['v'],p['u'],design,True)
p=s['panel_b'];B=fit(p['u'],p['v'],design);Bi=fit(p['v'],p['u'],design)
p=s['panel_c'];C=fit(p['u'],p['v1'],lambda x: design(np.sign(x)*(-np.expm1(-1.8*abs(x)))));D=fit(p['u'],p['v2'],lambda x:design(x,x*x/np.sqrt(1+.389*x*x)))
x=s['train']['x']; t=s['train']['traj']; y=s['train']['y']; T=np.c_[y,t]
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def quad(a): i,j=np.triu_indices(a.shape[1]);return a[:,i]*a[:,j]
def score(g, ridge):
 h=g[:52];d=np.linalg.norm(h,axis=0);d=np.where(d>1e-10,d,1);g=g/d; h=g[:52]; w=np.linalg.solve(h.T@h+np.diag(np.r_[1e-10,np.full(h.shape[1]-1,ridge)]),h.T@T[:52]); r=abs(g[52:]@w-T[52:]);return r[:,0].mean(),r[:,1:].mean()
roots={'x':x,'A':A(x),'Ai':Ai(x),'B':B(x),'Bi':Bi(x),'BA':B(A(x)),'BAi':B(Ai(x)),'BiA':Bi(A(x)),'BiAi':Bi(Ai(x)),'AB':A(B(x)),'AiB':Ai(B(x))}; out=[]
for nm,a in roots.items():
 c,d=C(a),D(a)
 opts={'lin':design(x),'poly':design(x,quad(a)), 'c':design(x,c),'d':design(x,d),'cd':design(x,c,d),'qc':design(x,c,quad(c)), 'qd':design(x,d,quad(d)), 'cdX':design(x,c,d,cross(c,d)),'cX':design(x,c,cross(a,c)), 'dX':design(x,d,cross(a,d)), 'cdQ':design(x,c,d,quad(a),quad(c),quad(d)), 'all':design(x,c,d,quad(a),quad(c),quad(d),cross(a,c),cross(a,d),cross(c,d)),'cD':design(x,c,D(c),cross(c,D(c))), 'dC':design(x,d,C(d),cross(d,C(d)))}
 for name,g in opts.items():
  for r in (1e-7,1e-4,.003,.03):out.append((*score(g,r),nm,name,r))
for idx in (0,1,2):
 print('leaders',idx);print('\n'.join(str(a) for a in sorted(out,key=lambda v:v[idx] if idx<2 else max(v[:2]))[:18]))
