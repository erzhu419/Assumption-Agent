import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']; tele=s['telemetry_vec']; np.set_printoptions(precision=6,suppress=True,linewidth=180)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def bias(x):return np.c_[np.ones(len(x)),x]
def amat(z,j):
 p=z[:,j]; q=z[:,(j+1)%4]; return np.c_[p*0+1,p,q,p*q,p**3]
a=s['panel_a']; ix=np.arange(48)
for _ in range(5):
 A=np.array([ls(amat(a['u'][ix],j),a['v'][ix,j]) for j in range(4)]); res=np.linalg.norm(np.column_stack([amat(a['u'],j)@A[j] for j in range(4)])-a['v'],axis=1); ix=np.argsort(res)[:45]
print('A',A,'residual',np.sort(res))
def af(z):return np.column_stack([amat(z,j)@A[j] for j in range(4)])
def ai(z):return least_squares(lambda u:(af(u.reshape(-1,4))-z).ravel(),z.ravel(),max_nfev=25).x.reshape(-1,4)
B=ls(bias(s['panel_b']['u']),s['panel_b']['v']);
def bf(x):return bias(x)@B
def bi(x):return (x-B[0])@np.linalg.inv(B[1:])
c=s['panel_c']; u=c['u']; p=np.stack(list(u.T)+[u[:,i]+sign*u[:,j] for i in range(4) for j in range(i+1,4) for sign in [-1,1]],axis=1); base=PolynomialFeatures(2).fit_transform(u); print('C quadratic RMS',np.std(c['v1']-base@ls(base,c['v1']),axis=0),np.std(c['v2']-base@ls(base,c['v2']),axis=0));
for label in ['v1','v2']:
 for j in range(4):
  r=c[label][:,j]; F=np.c_[bias(u),u*u,u*np.roll(u,-1,axis=1),u**3,np.sin(p),np.cos(p),np.tanh(p),np.abs(p),p*p,p**3]; M=bias(u); E=F-M@ls(M,F); v=r-M@ls(M,r); idx=np.argsort(abs(E.T@v)/(np.linalg.norm(E,axis=0)+1e-10))[-6:][::-1]; print(label,j,idx,np.round((abs(E.T@v)/(np.linalg.norm(E,axis=0)+1e-10))[idx]/np.linalg.norm(v),3))
Z={'x':x,'A':af(x),'Ai':ai(x),'B':bf(x),'Bi':bi(x),'AB':bf(af(x)),'AiB':bf(ai(x)),'ABi':bi(af(x)),'AiBi':bi(ai(x)),'BA':af(bf(x)),'BAi':ai(bf(x))}
for name,z in Z.items():
 cand=[]
 for d in [1,2,3,4]:
  f=PolynomialFeatures(d).fit_transform(z); Q=f@f.T
  for l in [1e-5,.001,.01,.1]:
   e=Q[52:,:52]@np.linalg.solve(Q[:52,:52]+l*np.eye(52),y[:52])-y[52:]; cand.append((np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),f'p{d}/{l}'))
 f=bias(z); b=ls(f[:52],y[:52]); print(name,'lin rmse',np.round(np.std(y[:52]-f[:52]@b,axis=0),3),'best',min(cand),min(cand,key=lambda q:q[1]))
