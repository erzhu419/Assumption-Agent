import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
aug=lambda x:np.c_[np.ones(len(x)),x]
rms=lambda x:np.mean(x*x)**.5
u=s['panel_a']['u'];v=s['panel_a']['v'];mask=np.linalg.norm(v-u,axis=1)<2
ab=np.array([ls(aug(u[mask,j:j+1]),v[mask,j]) for j in range(4)])
b=s['panel_b']; bb=ls(aug(b['u']),b['v'])
c=s['panel_c']; u=c['u']; z=c['v1']; w=c['v2']; np.set_printoptions(precision=4,suppress=True)
for name,fn,init in [('tanh',np.tanh,1.7),('sin',np.sin,1.3),('atan',np.arctan,2),('softsign',lambda x:x/(1+abs(x)),2)]:
 for shift in [False,True]:
  for skip in [False,True]:
   def feat(x,p):
    a=fn(x*p[:4]+(p[4:] if shift else 0));return aug(np.c_[a,x] if skip else a)
   def run(mask):
    def res(p):
     a=feat(u[mask],p);return (a@ls(a,z[mask])-z[mask]).ravel()
    p=least_squares(res,np.r_[np.repeat(init,4),np.zeros(4) if shift else []],max_nfev=200).x
    a=feat(u[mask],p);return p,ls(a,z[mask]),rms(res(p))
   p,m,e=run(np.ones(24,bool)); mm=np.arange(24)<18; pp,mmc,ee=run(mm); ev=rms(feat(u[~mm],pp)@mmc-z[~mm]);print('v1',name,shift,skip,e,ev,p)
   if e<.001:print(m)
for name,x in [('u',u),('v1',z),('tanh 1.7u',np.tanh(1.7*u)),('tanh 1.7v1',np.tanh(1.7*z)),('both',np.c_[u,z])]:
 q=PolynomialFeatures(2).fit_transform(x); a=ls(q,w);print('v2',name,rms(q@a-w),rms(q[18:]@ls(q[:18],w[:18])-w[18:]));
 if x.shape[1]==4:print('coefs',a)
X=s['train']['x'];Y=s['train']['y'];T=s['train']['traj']; tt=s['eval_times']; uu,ss,vv=np.linalg.svd(T,full_matrices=False); print('time',tt,'basis',vv[:2]); P=T@vv[:2].T
for name,x in [('raw',X),('cal',X*ab[:,1]+ab[:,0]),('inverse',(X-ab[:,0])/ab[:,1])]:
 bx=aug(x)@bb
 for name2,d in [('x',x),('Bx',bx),('tanhx',np.tanh(1.7*x)),('tanhBx',np.tanh(1.7*bx)),('sinx',np.sin(1.7*x)),('sinBx',np.sin(1.7*bx))]:
  for degree in [1,2]:
   a=PolynomialFeatures(degree).fit_transform(d);e=a[52:]@ls(a[:52],np.c_[Y,P][:52])-np.c_[Y,P][52:];print(name,name2,degree,np.mean(abs(e),axis=0))
