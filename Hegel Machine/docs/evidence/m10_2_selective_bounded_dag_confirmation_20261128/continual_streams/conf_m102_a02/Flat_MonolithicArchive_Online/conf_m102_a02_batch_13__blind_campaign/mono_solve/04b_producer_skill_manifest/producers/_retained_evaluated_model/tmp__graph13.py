import numpy as np,pickle,runpy,itertools
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
m=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a02_batch_06_solver.py'); x=m['_inverse'](tr['x'],m['_calibration'](s['panel_a'])); ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]; idx=np.triu_indices(4); quad=lambda x:np.c_[aff(x),x[:,idx[0]]*x[:,idx[1]]]
def panel(key,out,fun):
 u,v=s[key]['u'],s[key][out]; p=minimize_scalar(lambda p:np.mean((fun(u,p)@ls(fun(u,p),v)-v)**2),bounds=(1.6,3) if out=='v1' else (.8,1.7),method='bounded').x; b=ls(fun(u,p),v); print(key,out,p,np.sqrt(np.mean((fun(u,p)@b-v)**2)),b.round(3)); return lambda x:fun(x,p)@b
c1=panel('panel_c','v1',lambda x,p:np.c_[aff(x),np.tanh(p*x)]); c2=panel('panel_c','v2',lambda x,p:np.c_[aff(x),np.cos(p*x)]);
u,v=s['panel_b']['u'],s['panel_b']['v']; fs={'cosx':lambda x,p:np.cos(p*x),'cossum':lambda x,p:np.cos(p*(x+np.roll(x,-1,1))),'cosdiff':lambda x,p:np.cos(p*(x-np.roll(x,-1,1))),'cosopp':lambda x,p:np.cos(p*(x-np.roll(x,2,1)))[:,:2],'sinprod':lambda x,p:np.sin(p*x*np.roll(x,-1,1))}; bs={}
for k,f in fs.items():
 def d(x,p):return np.c_[aff(x),f(x,p)]
 def loss(p):return np.mean((d(u,p)@ls(d(u,p),v)-v)**2)
 p0=min(np.linspace(.15,6,120),key=loss);p=minimize_scalar(loss,bounds=(max(.03,p0-.1),p0+.1),method='bounded').x;b=ls(d(u,p),v);print(k,p,loss(p)**.5);bs[k]=lambda x,p=p,b=b,f=f:np.c_[aff(x),f(x,p)]@b
bs['bpoly']=lambda x:quad(x)@ls(quad(u),v)
ops={'C':c1,'D':c2,**bs}; bases={'x':x,'raw':tr['x']}; stages={**bases,**{k:f(x) for k,f in ops.items()},**{k+'raw':f(tr['x']) for k,f in ops.items()}}
for k,j in itertools.product(ops,repeat=2):stages[k+j]=ops[j](stages[k]);
h=tr['traj'][:,-1]; y=tr['y']; scores=[[],[]];
for k,z in stages.items():
 for k2,d in [('a',aff(z)),('q',quad(z)),('xq',np.c_[quad(x),quad(z)[:,1:]]),('xa',np.c_[aff(x),aff(z)[:,1:]])]:
  scale=np.maximum(np.linalg.norm(d[:n],axis=0),1e-10);D=d/scale
  for a in (0,.003,.03):
   yh=D[n:]@ls(np.r_[D[:n],np.sqrt(a)*np.eye(D.shape[1])],np.r_[np.c_[y,h][:n],np.zeros((D.shape[1],2))]); e=np.mean(abs(yh-np.c_[y,h][n:]),axis=0)
   for i in range(2):scores[i].append((e[i],k,k2,a))
for sc in scores:print(sorted(sc)[:27])
