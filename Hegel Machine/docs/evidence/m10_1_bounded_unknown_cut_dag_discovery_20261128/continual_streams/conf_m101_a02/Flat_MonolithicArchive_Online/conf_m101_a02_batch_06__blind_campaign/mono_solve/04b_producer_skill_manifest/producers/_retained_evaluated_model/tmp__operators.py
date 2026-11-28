import numpy as np,pickle
from itertools import combinations_with_replacement,combinations
from scipy.optimize import least_squares
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; XT=s['test']['x']; T=s['eval_times']
def fit(a,b):return lstsq(a,b)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
mon=[i for n in range(4) for i in combinations_with_replacement(range(4),n)]
def pol(x,d=3):return np.column_stack([np.prod(x[:,i],axis=1) for i in mon if len(i)<=d])
p=s['panel_a']; u,v=p['u'],p['v']; d=aff(u); ii=np.argsort(np.linalg.norm(v-d@fit(d,v),axis=1))[:-3]; cc=fit(pol(u[ii]),v[ii]); cc[abs(cc)<.001]=0; print('cal',[(m,np.round(c,4).tolist()) for m,c in zip(mon,cc) if np.any(c)],'rms',np.std(pol(u[ii])@cc-v[ii])); lin=fit(d[ii],v[ii]);
def inv(x):
 z=np.linalg.solve(lin[1:].T,(x-lin[0]).T).T
 for _ in range(12):
  h=1e-5; J=np.stack([(pol(z+h*e)-pol(z-h*e))@cc/(2*h) for e in np.eye(4)],axis=-1); step=np.linalg.solve(J,(pol(z)@cc-x)[...,None])[...,0]; z-=step
 print('inverse residual',np.max(abs(pol(z)@cc-x)), 'range',z.min(),z.max());return z
z=inv(np.r_[X,XT]);
def estimate(u,v,fn,p0,bounds):
 def resid(p):
  d=fn(u,p);return (d@fit(d,v)-v).ravel()
 p=least_squares(resid,p0,bounds=bounds,gtol=1e-12,ftol=1e-12,xtol=1e-12).x; a=fn(u,p); c=fit(a,v); h=(np.linalg.svd(a,full_matrices=False)[0]**2).sum(1); rms=np.sqrt(np.mean(resid(p)**2)); loo=np.mean(abs((a@c-v)/(1-h[:,None]))); return rms,loo,p,lambda z:fn(z,p)@c
ops={}
for name,panel,k in [('B','panel_b','v'),('C','panel_c','v1'),('D','panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][k]; cand=[]
 funcs=[('local',lambda x,p:aff(x*(1-p[1]*np.exp(-p[0]*x*x))),[1,1.1],([.1,.05],[5,4])),('local+',lambda x,p:np.c_[aff(x),x*np.exp(-p[0]*x*x)],[1],(.1,5)),('tanh',lambda x,p:aff(np.tanh(p[0]*x)),[1.5],(.2,5)),('tanh+',lambda x,p:np.c_[aff(x),np.tanh(p[0]*x)],[1.5],(.2,5)),('sin+',lambda x,p:np.c_[aff(x),np.sin(p[0]*x)],[2],(.2,5)),('cos+',lambda x,p:np.c_[aff(x),np.cos(p[0]*x)],[1],(.2,4)),('cos4+',lambda x,p:np.c_[aff(x),np.cos(p*x)],[1]*4,(.2,4)),('rational+',lambda x,p:np.c_[aff(x),x/(1+p[0]*x*x)],[1],(.05,10))]
 for key,fn,p0,bd in funcs:
  r,l,p,f=estimate(u,v,fn,p0,bd); cand.append((l,r,key,p,f))
 cand.sort(key=lambda x:x[0]);print(name,[(a,b,c,d.round(5)) for a,b,c,d,e in cand]);ops[name]=cand[0][-1]
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),(np.arange(20,72),np.arange(20))]
def scores(a):
 out=[]
 for i,j in folds:
  r=abs(a[j]@fit(a[i],Y[i])-Y[j]);out.append([r[:,0].mean(),r[:,1:].mean()]);
 return np.array(out)
if __name__=='__main__':
 coords={'z':z}; coords.update({k:f(z) for k,f in ops.items()});coords.update({k+l:f(coords[k]) for k in 'BCD' for l,f in ops.items()});
 results=[]
 for k,x in coords.items():
  for deg in (1,2,3): results.append((scores(pol(x,deg)),k+str(deg)))
 for k,l in combinations(coords,2):
  a,b=coords[k],coords[l]; x=np.c_[pol(a,2),pol(b,2)[:,1:]];results.append((scores(x),k+'+'+l))
 for target in (0,1):
  print('top',target,[(k,v.round(4).tolist()) for v,k in sorted(results,key=lambda p:p[0][:,target].mean())[:10]])
