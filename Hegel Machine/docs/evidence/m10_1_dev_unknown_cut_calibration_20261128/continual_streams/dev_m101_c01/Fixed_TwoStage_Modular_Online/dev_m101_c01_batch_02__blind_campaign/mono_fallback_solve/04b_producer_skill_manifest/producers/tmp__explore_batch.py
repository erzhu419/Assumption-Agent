import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z,t=d['x'],d['y'],d['traj'],s['eval_times']; np.set_printoptions(precision=5,suppress=True,linewidth=135)
print('times telemetry summary',t,s['telemetry_vec'],d['summary']); print('std',np.std(np.c_[x,y,z],0)); print('test ranges',np.min(s['test']['x'],0),np.max(s['test']['x'],0))
def poly(x,degree):
 n=x.shape[1]; ii=[a for k in range(degree+1) for a in it.combinations_with_replacement(range(n),k)]; return np.column_stack([np.prod(x[:,a],axis=1) for a in ii]),ii
u=s['panel_a']['u']; v=s['panel_a']['v']; P,ii=poly(u,1); keep=np.ones(len(u),bool)
for _ in range(5):
 c=np.linalg.lstsq(P[keep],v[keep],rcond=None)[0]; r=np.linalg.norm(v-P@c,axis=1); keep=r<4*np.median(r)
print('panel_a affine',sum(keep),c,'resid',np.sort(r)[::4]);
for name,key,keep in [('panel_a','v',keep),('panel_c','v1',np.ones(24,bool)),('panel_c','v2',np.ones(24,bool))]:
 u=s[name]['u'][keep];v=s[name][key][keep]; split=np.arange(len(u))%4!=0
 for degree in (1,2,3):
  P,ii=poly(u,degree); c=np.linalg.lstsq(P[split],v[split],rcond=None)[0]; print(name,key,degree,'rmse',np.sqrt(np.mean((v-P@c)**2,axis=0)),'held',np.sqrt(np.mean((v[~split]-P[~split]@c)**2,0)))
  if degree==2:
   c=np.linalg.lstsq(P,v,rcond=None)[0]; print([(i,a.round(3).tolist()) for i,a in zip(ii,c)])
 for funs in ([np.sin,np.cos],[np.sin,np.cos,np.tanh],[np.sin,np.cos,np.square]):
  for f in (1,2,3):
   P=np.c_[np.ones(len(u)),u,*[g(f*u) for g in funs]];c=np.linalg.lstsq(P[split],v[split],rcond=None)[0]; print(name,key,str([g.__name__ for g in funs]),f,np.sqrt(np.mean((v[~split]-P[~split]@c)**2)))

def ex(t,f):return np.exp(-t*f)
for name,fn,init,bounds in [
('saturating',lambda p:1-ex(t[:,None],p),[1.5,.2],([.01,.001],[20,20])),
('decay_sin_cos',lambda p: ex(t,p[0])[:,None]*np.c_[np.sin(t*p[1]),np.cos(t*p[1])],[.5,2],([-.5,.01],[10,20])),
('decay_sin_saturate',lambda p:np.c_[ex(t,p[0])*np.sin(t*p[1]),1-ex(t,p[2])],[.5,2,1],([-.5,.01,.005],[10,20,20])),
('damped_sin_cos1',lambda p:ex(t,p[0])[:,None]*np.c_[np.sin(t*p[1]),1-np.cos(t*p[1])],[.5,2],([-.5,.01],[10,20])),
('saturating_sine',lambda p: np.c_[np.sin(t*p[0]),1-ex(t,p[1])],[2,1],([.01,.005],[20,20])),
('both_damped_sines',lambda p: ex(t[:,None],p[:2])*np.sin(t[:,None]*p[2:]),[.5,.2,2,.5],([-.5,-.5,.01,.01],[10,10,20,20])),
('t_t2_decay',lambda p: ex(t,p[0])[:,None]*np.c_[t,t*t],[1],([-.5],[20]))]:
 def fun(p):
  a=fn(p); return (a@np.linalg.lstsq(a,z.T,rcond=None)[0]-z.T).ravel()
 fit=least_squares(fun,init,bounds=bounds); print(name,fit.x,np.mean(abs(fit.fun)),np.sqrt(np.mean(fit.fun**2)))
U,S,V=np.linalg.svd(z,full_matrices=False); print('temporal right vectors',V[:2]); print('rank2 noise',np.sqrt(np.mean(S[2:]**2)/z.size*len(S[2:]))); print('basis via extrapolation',np.linalg.lstsq(z[:,[0,-1]],z,rcond=None)[0]); print('amp fits')
Y=np.c_[y,U[:,:2]*S[:2]]
for degree in range(1,5):
 P,ii=poly(x,degree); c=np.linalg.lstsq(P[:56],Y[:56],rcond=None)[0]; print(degree,np.mean(abs(P[:56]@c-Y[:56]),0),np.mean(abs(P[56:]@c-Y[56:]),0));
 if degree==2: print('quadratic',list(zip(ii,c.round(3).tolist())))
