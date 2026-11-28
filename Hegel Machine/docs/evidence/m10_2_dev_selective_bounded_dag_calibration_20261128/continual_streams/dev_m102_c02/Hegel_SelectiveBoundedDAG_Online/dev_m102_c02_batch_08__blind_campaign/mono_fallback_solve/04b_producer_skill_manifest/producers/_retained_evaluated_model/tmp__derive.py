import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement as cr,combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def lin(z):return np.c_[np.ones(len(z)),z]
def ls(z,y):return np.linalg.lstsq(z,y,rcond=1e-10)[0]
def fit(z,y):return z@ls(z,y)
def rms(e):return np.mean(e**2)**.5
a=s['panel_a'];u=a['u'];v=a['v']; q=u*np.roll(u,-1,axis=1); F=np.c_[lin(u),q];r=np.linalg.norm(v-fit(F,v),axis=1); keep=np.argsort(r)[:-3];U=u[keep];V=v[keep];
def A(z,j):
 x=z[:,j]; w=z[:,(j+1)%4];return np.c_[np.ones(len(x)),x,w,x*w]
for j in range(4):
 x=U[:,j];w=U[:,(j+1)%4];y=V[:,j];F=A(U,j); print('A',j,ls(F,y),rms(y-fit(F,y)))
 z=U; D={'xw2':x*w*w,'x2w':x*x*w,'w2':w*w,'x2':x*x,'x3':x**3,'w3':w**3,'triple':np.prod(z[:,[j,(j+1)%4,(j+2)%4]],axis=1),'triple2':np.prod(z[:,[j,(j+1)%4,(j+3)%4]],axis=1)}
 for k in [1,1.25,2,3]:
  for nm,f in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('rat',lambda t:t/(1+t*t)),('sat',lambda t:1/(1+t*t))]:
   for l,t in [('x',x),('w',w),('xd',x-w),('xs',x+w),('other',z[:,(j+2)%4]),('other2',z[:,(j+3)%4]),('q',x*w)]:
    for pn,p in [('1',1),('x',x),('w',w),('q',x*w)]:D[f'{pn}*{nm}{k}({l})']=p*f(k*t)
 R=sorted((rms(y-fit(np.c_[F,d],y)),k) for k,d in D.items());print(R[:10]);print('res',np.c_[U,y-fit(F,y)][:6])
for name in ['b','c']:
 p=s['panel_'+name];x=p['u'];
 for n,y in p.items():
  if n=='u':continue
  print(name,n,'range',y.min(0),y.max(0))
  FS={'linear':lambda z:z,'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'ratan':lambda z:z/(1+z*z),'soft':lambda z:z/np.sqrt(1+z*z),'sinh':np.sinh}
  for nm,f in FS.items():
   for ex in (0,1):
    def des(k):return np.c_[lin(f(k*x)), x*x] if ex else lin(f(k*x))
    opt=minimize_scalar(lambda k:rms(y-fit(des(k),y)),bounds=(.02,3.5),method='bounded'); print(nm,ex,round(opt.x,3),round(opt.fun,5),end='; ')
   print()
  # Ridged index models per output, nonlinear functions of projected input plus constant/linear.
  for typ in ('tanh','sin','atan','rat','soft','relu','sinc'):
   funcs={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'soft':lambda z:z/np.sqrt(1+z*z),'relu':lambda z:np.maximum(0,z),'sinc':lambda z:np.sinc(z/np.pi)};f=funcs[typ]; scores=[]
   for j in range(4):
    w=ls(lin(x),y[:,j]); normalized=w/max(np.linalg.norm(w[1:]),1e-6)
    def ff(t):return np.c_[lin(x),f(lin(x)@t)]
    rr=min((least_squares(lambda t:fit(ff(t),y[:,j])-y[:,j],normalized*k,max_nfev=90) for k in [.6,1.25,2.5]),key=lambda s:s.cost);scores.append(rms(rr.fun))
   print(typ,np.round(scores,4))
t=s['train']; yy=t['y'];T=t['traj']; P=np.linalg.svd(T-T.mean(0),full_matrices=False)[2];print('modes',P[:4]);print('y vs traj',rms(yy-fit(lin(T@P[:3].T),yy))); print('X, y, temporal',np.c_[t['x'],yy,T@P[:3].T][:16]);
