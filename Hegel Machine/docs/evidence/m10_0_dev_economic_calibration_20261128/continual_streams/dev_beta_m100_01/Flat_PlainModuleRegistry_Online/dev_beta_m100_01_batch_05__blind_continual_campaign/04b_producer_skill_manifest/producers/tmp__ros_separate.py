import numpy as np
from scipy.optimize import least_squares
from ros_probe import x,y
from ros_joint_input import bank
bi=bank['bank_id']; z=bank['ref_signal']; sy=bank['raw_readout']; keep=np.max(np.abs(y),axis=1)<5; sb=np.abs(sy)<5; axis=np.count_nonzero(x,axis=1)==1; opp=[2,3,0,1]

def design(a,k,m):
 t=[a,np.sin(k[0]*a)]
 if 't' in m:t.append(np.tanh(k[1]*a))
 if '3' in m:t.append(a**3)
 if '5' in m:t.append(a**5)
 if 'h' in m:t.append(np.sin(2*k[0]*a))
 return np.stack(t,axis=-1)

def fit_local(tr,m):
 ii,jj=np.where((x!=0)&(tr&axis)[:,None]); aa=np.r_[x[ii,jj],z[sb]]; yy=np.r_[y[ii,jj],sy[sb]]; bb=np.r_[jj,bi[sb]]
 def get(p):
  g=np.r_[1,p[:3]]; D=np.c_[np.eye(4)[bb], g[bb,None]*design(aa,p[3:],m)]; b=np.linalg.lstsq(D,yy,rcond=None)[0]; return D@b-yy,b,g
 p=min((least_squares(lambda p:get(p)[0],[.64,.992,.65,2.5,k],bounds=([.25,.25,.25,1.5,.04],[2,2,2,3.5,3]),max_nfev=150) for k in (.2,.55,1.3)),key=lambda q:q.cost).x
 r,b,g=get(p); return p,b,g,np.sqrt(np.mean(r*r))

def cd(a,b,k,m):
 t=[b]
 if 'c' in m:t.append(1-np.cos(k[0]*b))
 if 'r' in m:t.append(b*b/(1+k[0]*b*b))
 if 's' in m:t.append(np.sin(k[1]*b))
 if 'q' in m:t.append(b*b)
 if '3' in m:t.append(b**3)
 if 'u' in m:t.append(b/(1+k[1]*b*b))
 if 'i' in m:t.extend([a*b,np.tanh(a)*b,a*(1-np.cos(2*b)),b*(1-np.cos(2*a))])
 if 'v' in m:t.extend([np.sin(2*(a+b))-np.sin(2*a)-np.sin(2*b),np.cos(2*(a+b))-np.cos(2*a)-np.cos(2*b)+1])
 return np.stack(t,axis=-1)
folds=[np.array([],int),*np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5)]
for lm in ('t','t3','t35','h'):
 p,b,g,e=fit_local(keep,lm);print('local',lm,e,p.round(6),b.round(6),flush=True)
for cm in ('c','r','qs','cs','rs','ru','r3','rsi','rsv'):
 errs=[];result=[]
 for ids in folds:
  tr=keep.copy();tr[ids]=False; va=ids[keep[ids]] if ids.size else np.flatnonzero(keep); p,b,g,_=fit_local(tr,'t3'); d=b[:4]+g*(design(x,p[3:],'t3')@b[4:]); R=(y-d)/g
  def cp(k):
   D=cd(x,x[:,opp],k,cm); B=np.linalg.lstsq(D[tr].reshape(-1,D.shape[-1]),R[tr].ravel(),rcond=None)[0];return D@B,B
  rng=([.15,.1],[4,6]); ini=[2 if 'c' in cm else .5,2.]
  opt=least_squares(lambda k:(cp(k)[0]-R)[tr].ravel(),ini,bounds=rng,max_nfev=120); rr,bb=cp(opt.x);errs.append(np.sqrt(np.mean(((rr[va]-R[va])*g)**2)));result.append((opt.x,bb))
 print(cm,np.round(errs,7),[np.round(a,6) for a in result[0]],flush=True)
