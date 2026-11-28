from b05_probe import *
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=5,suppress=True)
d=S['panel_b'];u=d['u'];v=d['v'];
def bank(x,p,kind):
 f=p[:4]; a=np.sin(x*f); c=np.cos(x*f)
 if kind=='linked':return np.c_[np.ones(len(x)),x+a*p[4:8]]
 if kind=='phase':return np.c_[np.ones(len(x)),x+a*p[4:8]+(c-1)*p[8:12]]
 if kind=='sine':return np.c_[np.ones(len(x)),x,a]
 if kind=='both':return np.c_[np.ones(len(x)),x,a,c]
 if kind=='cube':return np.c_[np.ones(len(x)),x,a,x**3]
mods={}
for kind in ['linked','phase','sine','both','cube']:
 p0=np.r_[[2.85]*4,[-.4]*4,[0.]*4] if kind=='phase' else np.r_[[2.85]*4,[-.4]*4] if kind=='linked' else np.full(4,2.85)
 def res(p):
  A=bank(u,p,kind); return (A@fit(A,v)-v).ravel()
 p=least_squares(res,p0,ftol=1e-11).x;A=bank(u,p,kind);c=fit(A,v); print(kind,'parameters',p,'fit',np.sqrt(np.mean(res(p)**2)),'LOO conditional',cvfit(A,v)); print('coefficients',c); mods[kind]=bank(za,p,kind)@c
T=tr['traj'];t=S['eval_times']; i=np.arange(52); j=np.arange(52,72)
def time(r):return -np.expm1(-t[:,None]*r)
def timefit(r):
 A=time(r);return (A@fit(A,T[i].T)-T[i].T).ravel()
r=least_squares(timefit,[1.7,.2],bounds=([.6,.01],[5.,.7])).x; B=time(r); amp=fit(B,T.T).T
print('rates',r,'RMS',np.std(timefit(r)), 'amp RMS',amp.std(0)); print('telemetry time proposals',[(q,np.std(timefit(q))) for q in ([1.7,.2],[1.7,.1],[1.8,.2],[1.8,.3],[1.8,.4],r)])
Y=np.c_[tr['y'],amp]
for k,z in mods.items():
 A=poly(z,2)[0];beta=fit(A[i],Y[i]);print('readout',k)
 for Q in (A[:,:5],A,np.c_[A[:,:5],z*z],np.c_[A[:,:5],z,np.sin(z),np.tanh(z)]):
  b=fit(Q[i],Y[i]);p=Q[j]@b;print('mae',np.mean(abs(p-Y[j]),axis=0),'traj',np.mean(abs(p[:,1:]@B.T-T[j])),'train',np.sqrt(np.mean((Q[i]@b-Y[i])**2,axis=0)))
 print('quad coef',beta.T)
 for h in range(3):
  q,rr=np.linalg.qr(A[i,:5]);base=Y[i,h]-q@(q.T@Y[i,h]); P=A[i,5:]-q@(q.T@A[i,5:]); ids=[]
  for _ in range(4):
   l=np.argmax(abs(P.T@base)/np.maximum(np.linalg.norm(P,axis=0),1e-12));ids.append(l); Q=A[:,np.r_[np.arange(5),5+np.array(ids)]]; b=fit(Q[i],Y[i,h]);base=Y[i,h]-Q[i]@b;print('sparse',h,ids,np.mean(abs(Q[j]@b-Y[j,h])),np.std(base))
