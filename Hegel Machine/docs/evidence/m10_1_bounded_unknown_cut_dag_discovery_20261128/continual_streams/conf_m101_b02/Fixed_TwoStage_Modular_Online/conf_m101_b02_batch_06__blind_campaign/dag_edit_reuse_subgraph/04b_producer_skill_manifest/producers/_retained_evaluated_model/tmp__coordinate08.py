import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
one=lambda x:np.ones((len(x),1))
pairs=np.array(list(combinations(range(4),2))).T
cross=lambda x:x[:,pairs[0]]*x[:,pairs[1]]
lin=lambda x:np.c_[one(x),x]
quad=lambda x:np.c_[lin(x),x*x,cross(x)]
def loo(a,y):
 p=np.linalg.pinv(a);return (y-a@(p@y))/(1-np.sum(a*p.T,axis=1)[:,None])
pa=s['panel_a'];u=pa['u'];v=pa['v'];a=lin(u);w=np.ones(len(u))
for _ in range(10):
 r=np.linalg.norm(v-a@fit(a*w[:,None],v*w[:,None]),axis=1); w=1/np.sqrt(1+(2*r/np.median(r))**2)
k=np.argsort(r)[:-3];u,v=u[k],v[k];print('A full cubic coefficients');print(np.round(fit(np.c_[quad(u),u**3],v),4))
def ab(x,j):
 h=x[:,j]; q=x[:,(j+1)%4];return np.c_[one(x),h,q,h*q,h**3,q**3]
ac=np.array([fit(ab(u,j),v[:,j]) for j in range(4)]);print('A cyclic coefficients',np.round(ac,5));print('A cyclic residual',np.sqrt(np.mean([(ab(u,j)@ac[j]-v[:,j])**2 for j in range(4)])))
def af(x):return np.column_stack([ab(x,j)@ac[j] for j in range(4)])
X=np.r_[s['train']['x'],s['test']['x']];z=np.stack([least_squares(lambda t:af(t[None])[0]-x,x).x for x in X]);print('inverse residual, ranges',np.max(abs(af(z)-X)),z.min(),z.max())
def assay(p,key,f,x):
 p=s[p];return f(x)@fit(f(p['u']),p[key])
c1=lambda x:np.c_[one(x),cross(x)[:,[0,2,3,5]]]
u=s['panel_c']['u'];v=s['panel_c']['v2'];o=minimize_scalar(lambda w:np.mean(loo(np.c_[lin(u),np.cos(w*u)],v)**2),bounds=(.5,2.5),method='bounded');print('C2 freq',o.x,'loo',o.fun**.5);c2=lambda x:np.c_[lin(x),np.cos(o.x*x)]
u=s['panel_b']['u'];v=s['panel_b']['v'];
for name,fn in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan)]:
 o=minimize_scalar(lambda w:np.mean(loo(np.c_[lin(u),fn(w*u)],v)**2),bounds=(.2,3.6),method='bounded');print('B',name,o.x,o.fun**.5,np.round(fit(np.c_[lin(u),fn(o.x*u)],v),3))
bf=lambda x:np.c_[lin(x),np.sin(2.8*x)]
Y=np.c_[s['train']['y'],s['train']['traj']];n=len(Y); folds=[np.arange(52,72),*np.array_split(np.random.default_rng(675).permutation(n),5)]
def stats(F):
 vals=[]
 for va in folds:
  tr=np.setdiff1d(np.arange(n),va);g=F[tr];p=F[va]@np.linalg.solve(g.T@g+.0003*np.eye(g.shape[1]),g.T@Y[tr]);e=abs(Y[va]-p);vals.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(vals)
for nm,x in [('raw',X),('inverse',z)]:
 b=assay('panel_b','v',bf,x);c=assay('panel_c','v1',c1,x);d=assay('panel_c','v2',c2,x)
 f={'b':b,'c':c,'d':d,'bc':b*c,'bd':b*d,'cd':c*d,'xb':x*b,'xc':x*c,'xd':x*d,'cs':c*c,'bs':b*b,'ds':d*d,'cube':x**3,'sin':np.sin(x),'tanh':np.tanh(x),'sin2':np.sin(2*x),'cos':np.cos(x)}
 for bn,g in [('lin',lin(x)),('quad',quad(x))]:
  opts=[]
  for sz in range(4):
   for ks in combinations(f,sz):
    v=stats(np.column_stack([g]+[f[k] for k in ks])); opts.append((max(v.max(0)),ks,v[0],v.max(0)))
  print(nm,bn,'baseline',opts[0]);print('best',sorted(opts,key=lambda p:p[0])[:7])
