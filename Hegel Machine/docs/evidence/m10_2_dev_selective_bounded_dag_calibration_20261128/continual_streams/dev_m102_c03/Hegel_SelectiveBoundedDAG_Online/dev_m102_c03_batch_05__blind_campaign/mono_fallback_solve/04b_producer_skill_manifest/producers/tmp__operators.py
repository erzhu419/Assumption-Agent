import numpy as np
from scipy.optimize import least_squares
from itertools import combinations

def one(x):return np.column_stack((np.ones(len(x)),x))
def fit(h,v):return np.linalg.lstsq(h,v,rcond=None)[0]
def calmat(z,j):
 u=z[:,j];v=z[:,(j+1)%4];return one(np.c_[u,v,u*v,u**3])
def cross(u):return one(u*np.roll(u,-1,axis=1))
def operators(s):
 p=s['panel_a'];u=p['u'];v=p['v'];h=one(u);r=np.linalg.norm(h@fit(h,v)-v,axis=1); good=r<np.median(r)*4; ca=np.stack([fit(calmat(u[good],j),v[good,j]) for j in range(4)])
 def inv(x):
  z=(x-ca[:,0])/ca[:,1];i=np.arange(len(x));j=np.arange(4);k=(j+1)%4
  for _ in range(10):
   w=np.roll(z,-1,axis=1);r=ca[:,0]+ca[:,1]*z+ca[:,2]*w+ca[:,3]*z*w+ca[:,4]*z**3-x
   d=np.zeros((len(x),4,4));d[:,j,j]=ca[:,1]+ca[:,3]*w+3*ca[:,4]*z**2;d[:,j,k]=ca[:,2]+ca[:,3]*z
   z-=np.linalg.solve(d,r[...,None])[...,0]
  return z
 p=s['panel_b']; b=fit(one(p['u']),p['v']); B=lambda u:one(u)@b
 p=s['panel_c'];u=p['u'];v=p['v2'];c=fit(cross(u),p['v1']); C=lambda u:cross(u)@c
 best=None
 for name,F in [('cos',lambda u,g:np.cos(u*g)),('sin2',lambda u,g:np.sin(u*u*g)),('tanh2',lambda u,g:np.tanh(u*u*g)),('rat',lambda u,g:u*u/(1+g*u*u)),('exp',lambda u,g:np.exp(-g*u*u))]:
  for g0 in [.2,.7,1.4,2.5,4.]:
   def err(g):
    h=one(F(u,g));return (h@fit(h,v)-v).ravel()
   g=least_squares(err,[g0],bounds=(.00001,8)).x;e=np.std(err(g));
   if best is None or e<best[0]:best=(e,name,F,g)
  print(name,g,e) if __name__=='__main__' else None
 e,name,F,g=best; vb=fit(one(F(u,g)),v);D=lambda u:one(F(u,g))@vb
 if __name__=='__main__': print('cal',ca,'panel rms',e,name,g,'C',c,'D',vb)
 return inv,B,C,D

if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; inv,B,C,D=operators(s)
 x=s['train']['x'];T=s['train']['traj'];y=s['train']['y'];t=s['eval_times']; z=inv(x)
 def q(p):return -np.expm1(-t[:,None]*p)
 p=least_squares(lambda p:(q(p)@fit(q(p),T[:52].T)-T[:52].T).ravel(),[.4,1.8]).x
 Y=np.c_[y,fit(q(p),T.T).T]
 fr={'z':z,'Bz':B(z),'Cz':C(z),'Dz':D(z),'CBz':C(B(z)),'DBz':D(B(z)),'BCz':B(C(z)),'BDz':B(D(z)),'Craw':C(x),'Draw':D(x),'CBraw':C(B(x)),'DBraw':D(B(x))}
 for k,u in fr.items():
  U=one(u);R=U[52:]@fit(U[:52],Y[:52])-Y[52:];print(k,'linear',np.mean(abs(R),0).round(4),'st',np.mean(abs(R[:,1:]@q(p).T)).round(4))
 keys=list(fr)
 for j in range(3):
  out=[]
  for n in range(1,4):
   for kk in combinations(keys,n):
    U=one(np.column_stack([fr[k] for k in kk])); w=fit(U[:52],Y[:52,j]);r=U@w-Y[:,j];out.append((np.mean(abs(r[52:])),np.mean(abs(r[:52])),kk))
  print('BEST LINEAR',j,sorted(out)[:8])
