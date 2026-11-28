from cal06 import *
from scipy.optimize import root, minimize_scalar
from itertools import combinations

def aff(x): return np.c_[np.ones(len(x)),x]
def geom(s):
 _,coef,err=calibrate(); u,v=s['panel_a'].values(); f=calterms(u); keep=err<.002
 coef=np.array([ls(f[keep,j],v[keep,j]) for j in range(4)])
 X=np.vstack((s['train']['x'],s['test']['x']))
 sol=[root(lambda z: np.sum(calterms(z)*coef,axis=-1)-x,x,tol=1e-10) for x in X]
 z=np.stack([a.x for a in sol]);print('inverse max',max(np.linalg.norm(a.fun) for a in sol))
 p=s['panel_b']; w=aff(z)@ls(aff(p['u']),p['v']); p=s['panel_c']; u=p['u']
 def pair(x):return x[:,[0,0,1,2]]*x[:,[1,3,2,3]]
 a=ls(aff(pair(u)),p['v1'])
 def err(k):
  F=aff(np.cos(k*u));return np.mean((F@ls(F,p['v2'])-p['v2'])**2)
 k=minimize_scalar(err,bounds=(2,3),method='bounded').x
 b=ls(aff(np.cos(k*u)),p['v2']);print('C cosine',k,err(k)**.5, 'v1 rms',np.mean((aff(pair(u))@a-p['v1'])**2)**.5)
 return z,w,[(aff(pair(x))@a,aff(np.cos(k*x))@b) for x in [z,w]]

if __name__=='__main__':
 z,w,base=geom(s); n=len(s['train']['x']); y=s['train']['y']; S=s['train']['traj']; t=s['eval_times']; T=lambda k:-np.expm1(-t[:,None]*k)
 def profile(k,Q):
  F=T(k);return (ls(F,Q.T).T@F.T-Q).ravel()
 p=least_squares(lambda k:profile(k,S[:56]),[.4,1.8,5],bounds=([.05,.7,3],[.7,3,12])).x; B=T(p); C=ls(B,S.T).T
 print('rates',p,'temporal residual',np.std(profile(p,S[:56])),np.std(profile(p,S[56:])))
 for i,(a,b) in enumerate(base):
  x=[z,w][i]; dot=(a*b).sum(1); ab=(a[:,:,None]*b[:,None,:]).reshape(-1,16); aa=np.array([a[:,j]*a[:,k] for j,k in combinations(range(4),2)]).T;bb=np.array([b[:,j]*b[:,k] for j,k in combinations(range(4),2)]).T
  F=np.c_[x,a,b,dot,ab[:,1],ab,aa,bb,a*a,b*b]; labels=['z'+str(j) for j in range(4)]+['a'+str(j) for j in range(4)]+['b'+str(j) for j in range(4)]+['dot','a0b1']+['a%db%d'%(j,k) for j in range(4) for k in range(4)]+['aa'+str(j) for j in range(6)]+['bb'+str(j) for j in range(6)]+['sq'+str(j) for j in range(8)]
  print('branch input',i)
  for name,H in [('z',x),('ab',F[:,4:12]),('ab+mix',F[:,4:14]),('z+ab+mix',F[:,:14]),('all_ab',F[:,4:30]),('all',F)]:
   H=aff(H); A=ls(H[:56],np.c_[y,C][:56]); pred=H[56:n]@A;print(name,'y/trajectory MAE',round(np.mean(abs(pred[:,0]-y[56:])),5),round(np.mean(abs(pred[:,1:]@B.T-S[56:])),5),'mode RMSE',np.round(np.mean((H[:56]@A-np.c_[y,C][:56])**2,0)**.5,5))
  for target in np.c_[y,C].T:
   sel=[]; H0=aff(F); H=H0[:56]; Yn=target[:56]; left=list(range(1,H.shape[1]));
   for j in range(9):
    def fit(cols):
     D=H[:,[0]+cols];c=ls(D,Yn);return np.linalg.norm(D@c-Yn),c
    q=min(left,key=lambda l:fit(sel+[l])[0]);left.remove(q);sel.append(q)
    if j in [3,5,8]:
     e,coef=fit(sel); print('sparse',j+1,[labels[k-1] for k in sel], 'train rmse',round(e/56**.5,5),'valid mae',round(np.mean(abs(H0[56:n][:,[0]+sel]@coef-target[56:])),5))
