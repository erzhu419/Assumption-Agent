from b05_probe import *
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement

def get_basis(x,f,kind):
 A=poly(x,1)[0]
 if kind=='sin': return np.c_[A,np.sin(x*f)]
 if kind=='cos': return np.c_[A,np.cos(x*f)]
 if kind=='both': return np.c_[A,np.sin(x*f),np.cos(x*f)]
 if kind=='pair': return np.c_[A,np.sin(x*f),np.sin((x+np.roll(x,-1,axis=1))*f),np.sin((x-np.roll(x,-1,axis=1))*f)]
 if kind=='quadsin': return np.c_[poly(x,2)[0],np.sin(x*f)]
 if kind=='quadcos': return np.c_[poly(x,2)[0],np.cos(x*f)]
 if kind=='cubic': return np.c_[A,x**3]

models={}
for key,name in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 d=S[key];x,y=d['u'],d[name];res=[]
 for kind in ('sin','cos','both','pair','quadsin','quadcos','cubic'):
  def objective(f):return cvfit(get_basis(x,f,kind),y)
  grid=np.linspace(.3,7,45); i=np.argmin([objective(f) for f in grid]);opt=minimize_scalar(objective,bounds=(max(.15,grid[i]-.18),grid[i]+.18),method='bounded');f=opt.x
  p=get_basis(x,f,kind); c=fit(p,y);res.append((opt.fun,kind,f,c,np.sqrt(np.mean((p@c-y)**2))))
 res.sort(key=lambda a:a[0]); print(key,name,[(a[1],round(a[2],5),round(a[0],7),round(a[4],7)) for a in res]);_,kind,f,c,_=res[0];print('coef',np.round(c,4));models[name]=lambda u,k=kind,f=f,c=c:get_basis(u,f,k)@c

if __name__=='__main__':
 x=np.r_[tr['x'],S['test']['x']]; basis=np.linalg.svd(tr['traj'][:52],full_matrices=False)[2][:2].T; Y=np.c_[tr['y'],tr['traj']@basis];
 def ev(P):
  pred=P[52:n]@fit(P[:52],Y[:52]);return np.round([np.mean(abs(pred[:,0]-Y[52:,0])),np.mean(abs(pred[:,1:]@basis.T-tr['traj'][52:]))],5)
 for label,z in [('raw',x),('cal',za),('a_forward',np.c_[x,np.roll(x,-1,axis=1)] if False else np.array([.1298,-.1352,-.1118,.0478])+x*np.array([.9964,.9219,.9825,.8549])-.32*np.roll(x,-1,axis=1)+.32*x*np.roll(x,-1,axis=1)+.16*x**3)]:
  features={label:z};features.update({k:g(z) for k,g in models.items()}); features.update({k+'(b)':g(features['v']) for k,g in models.items() if k!='v'});features['v(v1)']=models['v'](features['v1']);features['v(v2)']=models['v'](features['v2']);
  for k,u in features.items():
   a=poly(u,1)[0];q=poly(u,2)[0];c=np.c_[a,u**3];print(label,k,'L/Q/C',ev(a),ev(q),ev(c),'addto',ev(np.c_[poly(z,1)[0],u]))
  for f in (1.,1.7,2.2,3.,3.8,4.5,5.5):
   a=poly(z,1)[0]; print(label,'sin',f,ev(np.c_[a,np.sin(f*z)]),ev(np.c_[a,np.sin(f*z),np.cos(f*z)]), 'cuberoot',ev(np.c_[a,np.cbrt(z)]))
 print('times',S['eval_times'])
