import numpy as np
from scipy.optimize import least_squares

def D(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]

def calibrate(panel,x):
 u,v=panel['u'],panel['v']; l=D(u); rng=np.random.default_rng(45)
 a=min((ls(l[i],v[i]) for i in (rng.choice(len(u),8,False) for _ in range(240))),key=lambda a:np.median(np.linalg.norm(l@a-v,axis=1)))
 r=np.linalg.norm(l@a-v,axis=1); good=r<5*np.median(r)
 a=ls(l[good],v[good]); w=np.roll(u,-1,axis=1)
 b,d,e,f,g=np.stack([ls(D(np.c_[u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3])[good],v[good,j]) for j in range(4)],axis=1)
 z=np.linalg.solve(a[1:].T,(x-a[0]).T).T; j=np.arange(4)
 for _ in range(20):
  w=np.roll(z,-1,axis=1); r=b+d*z+e*w+f*z*w+g*z**3-x
  if not r.size or abs(r).max()<1e-12: break
  J=np.zeros((len(z),4,4)); J[:,j,j]=d+f*w+3*g*z*z; J[:,j,(j+1)%4]=e+f*z
  z-=np.linalg.solve(J,r[...,None])[...,0]
 return z

FUN={'softsign':lambda x:x/(1+abs(x)),'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'rat':lambda x:x/(1+x*x),'softplus':lambda x:np.logaddexp(0,x),'silu':lambda x:x/(1+np.exp(-x)),'gauss':lambda x:np.exp(-x*x),'erf':__import__('scipy.special',fromlist=['erf']).erf,'arctan':np.arctan}
def branch(u,v,z,name,cat=1,shift=1,k0=1.2):
 fun=FUN[name]
 def mat(a,p):
  q=fun(a*p[0]+(p[1:] if shift else 0)); return D(np.c_[a,q] if cat else q)
 def res(p):
  H=mat(u,p); return (H@ls(H,v)-v).ravel()
 p=least_squares(res,[k0]+[0.]*(4*shift),bounds=([.04]+[-5.]*(4*shift),[7.]+[5.]*(4*shift)),max_nfev=160).x
 h=mat(u,p); return np.std(res(p)),p,mat(z,p)@ls(h,v),mat(z,p)

def timebasis(t,Y):
 def h(p): return -np.expm1(-t[:,None]*p)
 def loss(p):
  H=h(p); return (H@ls(H,Y.T)-Y.T).ravel()
 return h(least_squares(loss,[1.8,.4],bounds=([.8,.025],[6.,.79])).x)

if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; tr=s['train']; z=calibrate(s['panel_a'],tr['x']); b=D(z)@ls(D(s['panel_b']['u']),s['panel_b']['v']); print('z range',z.min(0),z.max(0));
 t=timebasis(s['eval_times'],tr['traj'][:52]); Y=np.c_[tr['y'],ls(t,tr['traj'].T).T]
 def cv(f):
  f=D(f); R=f[52:]@ls(f[:52],Y[:52])-Y[52:]; return np.round(np.r_[np.mean(abs(R),0),np.mean(abs(R[:,1:]@t.T))],5)
 def pair(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
 cand={}
 for k in ['v1','v2']:
  v=s['panel_c'][k]; L=[]
  for name in FUN:
   for cat in [0,1]:
    for shift in [0,1]:
     for k0 in ([1.,2.4,3.5] if name in ['sin','cos'] else [1.2]):
      a=branch(u,v,z,name,cat,shift,k0); L.append((*a[:2],name,cat,shift,k0,a[2],a[3]))
  L.sort(key=lambda a:a[0]); cand[k]=L; print(k,[(round(a[0],7),a[1].round(4),a[2:6]) for a in L[:10]])
 for j in [0,1,2,3,5,8]:
  a=cand['v1'][j][-2]; c=cand['v2'][0][-2]; cross=pair(a,c); dot=(a*c).sum(1)
  for name,f in [('a,c,dot',np.c_[a,c,dot]),('a,c,dot,c01',np.c_[a,c,dot,a[:,0]*c[:,1]]),('a,c,cross',np.c_[a,c,cross]),('z,a,c,cross',np.c_[z,a,c,cross]),('b,a,c,dot',np.c_[b,a,c,dot]),('a,c,all2',np.c_[a,c,pair(a,a),cross,pair(c,c)])]: print(j,name,cv(f))
 for q in [z,b]:
  for a in [q,cand['v1'][0][-2],cand['v2'][0][-2]]:
   print('base',cv(np.c_[q,a,pair(q,a)]))
