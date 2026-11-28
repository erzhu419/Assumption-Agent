import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; T=tr['traj']; y=tr['y']; t=s['eval_times']; N=len(x); np.set_printoptions(precision=4,suppress=True,linewidth=145)
def affine(p):
 u,v=p['u'],p['v']; D=np.c_[np.ones(len(u)),u]; m=np.ones(len(u),bool)
 for _ in range(4):
  w=np.linalg.lstsq(D[m],v[m],rcond=None)[0]; e=np.linalg.norm(D@w-v,axis=1); m=e<4*np.median(e)+1e-9
 return w
A=affine(s['panel_a']); B=affine(s['panel_b']); za=(x-A[0])/np.diag(A[1:]); zb=(x-B[0])@np.linalg.inv(B[1:]); zba=(za-B[0])@np.linalg.inv(B[1:]); zab=(zb-A[0])/np.diag(A[1:]);
def poly(x,d):
 return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def timefit(T):
 def basis(p): return np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)]
 def fun(p):
  D=basis(p);return (T@D@np.linalg.pinv(D)-T).ravel()
 return basis(least_squares(fun,[1.55,.26],bounds=([.7,.05],[3.,.65])).x)

def ridgebranch(x,Y,ii,jj,f,start,kind):
 D=poly(x,2); L=poly(x,1); R=D[ii]; Z=Y[ii]; QR=np.linalg.pinv(R)
 if kind=='ridge':
  def fun(p):
   g=f(L[ii]@p); h=g-R@(QR@g); q=Z-R@(QR@Z); a=h@q/max(h@h,1e-18); return q-a*h
  r=least_squares(fun,start,max_nfev=150,bounds=(-6.,6.),gtol=1e-9); G=np.c_[D,f(L@r.x)]
 elif kind=='outer':
  def fun(p):
   g=f(D[ii]@p); q=np.c_[np.ones(len(ii)),g]; return q@np.linalg.lstsq(q,Z,rcond=None)[0]-Z
  p=np.linalg.lstsq(D[ii],Z,rcond=None)[0]*start; r=least_squares(fun,p,max_nfev=120); G=np.c_[np.ones(len(x)),f(D@r.x)]
 w=np.linalg.lstsq(G[ii],Z,rcond=None)[0]; return np.mean(abs(G[jj]@w-Y[jj])),np.mean(abs(G[ii]@w-Z)),r.x,w
if __name__=='__main__':
 ii=np.arange(56); jj=np.arange(56,72); D=timefit(T[ii]); F=T@np.linalg.pinv(D.T); print('temporal basis',D,'amplitudes',F.std(0),'rms',np.std(T-F@D.T,axis=0)); targets=np.c_[y,F];
 for name,z in [('raw',x),('Ainv',za),('Binv',zb),('AinvBinv',zba),('BinvAinv',zab),('AinvBforward',za@B[1:]+B[0])]:
  Q=poly(z,2); w=np.linalg.lstsq(Q[ii],targets[ii],rcond=None)[0]; print(name,'polynomial coefficients',w.T)
 for k,Y in enumerate(targets.T):
  D=poly(za,2); p=np.linalg.lstsq(D[ii],Y[ii],rcond=None)[0][1:5]; H=np.zeros((4,4)); w=np.linalg.lstsq(D[ii],Y[ii],rcond=None)[0];
  for c,(i,j) in zip(w[5:],combinations_with_replacement(range(4),2)): H[i,j]=H[j,i]=c if i==j else c/2
  _,V=np.linalg.eigh(H); starts=[np.r_[b,v] for b in [0,.5] for v in [p/np.linalg.norm(p),*V.T,*np.eye(4)]]
  print('TARGET',k)
  for fn,f in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('rational',lambda x:x/(1+x*x)),('exp',lambda x:np.exp(np.clip(x,-20,20)))]:
   rr=sorted([ridgebranch(za,Y,ii,jj,f,st,'ridge') for st in starts],key=lambda v:v[1]); print(fn,[(round(a,5),round(b,5),c,d[-1]) for a,b,c,d in rr[:2]])
  for fn,f in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('rational',lambda x:x/(1+x*x))]:
   rr=sorted([ridgebranch(za,Y,ii,jj,f,st,'outer') for st in [.25,.65,1,1.5]],key=lambda v:v[1]); a,b,c,d=rr[0]; print('outer',fn,round(a,5),round(b,5),c[:5],d)
