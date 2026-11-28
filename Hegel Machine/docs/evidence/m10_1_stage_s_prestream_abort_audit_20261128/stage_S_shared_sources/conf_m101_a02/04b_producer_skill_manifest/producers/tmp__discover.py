import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; y=tr['y']; Z=tr['traj']; t=s['eval_times']; np.set_printoptions(precision=6,suppress=True,linewidth=145)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,z):return np.linalg.lstsq(x,z,rcond=None)[0]
a=s['panel_a']; U=aff(a['u']);V=a['v']; ix=np.arange(len(U))
for _ in range(5):
 W=ls(U[ix],V[ix]);ix=np.argsort(np.linalg.norm(U@W-V,axis=1))[:45]
cal=np.array([ls(aff(a['u'][ix,k:k+1]),V[ix,k]) for k in range(4)])
b=s['panel_b']; B=ls(aff(b['u']),b['v']); u,sv,vh=np.linalg.svd(B[1:]); B[1:]=sv.mean()*(u@vh); B[0]=(b['v']-b['u']@B[1:]).mean(0)
c=(X-cal[:,0])/cal[:,1]; z=aff(c)@B; F=aff(z); H=ls(F,Z)
def temp(p):return -np.expm1(-np.outer(t,p))
def amplitudes(i):
 h=ls(F[i],Z[i]); r=least_squares(lambda p:(F[i]@h@np.linalg.pinv(temp(p).T)@temp(p).T-F[i]@h).ravel(),[.42,1.8]); T=temp(r.x); return h@np.linalg.pinv(T.T),r.x
p=z[:,0]*z[:,2]; q=z[:,1]*z[:,3]; P=np.c_[F,*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]]
fs={'full':P,'pair':np.c_[F,p,q],'tied_pair':np.c_[F,p+q],'tied_linear':np.c_[np.ones(len(z)),z[:,0]+z[:,1],z[:,2]+z[:,3],p,q],'tied_both':np.c_[np.ones(len(z)),z[:,0]+z[:,1],z[:,2]+z[:,3],p+q],'tied_sign':np.c_[np.ones(len(z)),-z[:,0]-z[:,1]+z[:,2]+z[:,3],p+q]}
print('B',B,'amplitudes in B/latent basis',amplitudes(np.arange(len(X))),sep='\n');print('latent amplitudes',ls(aff(c), F@amplitudes(np.arange(len(X)))[0]));
for k,f in fs.items():
 w=ls(f,y); r=y-f@w; print(k,w,'RMS',np.sqrt(np.mean(r*r)),'SE',np.sqrt(np.diag(np.linalg.inv(f.T@f))*(r@r)/(len(y)-len(w))))
edge=np.linalg.norm((X-X.mean(0))/X.std(0),axis=1); rng=np.random.default_rng(3); folds=[('tail',np.arange(56),np.arange(56,72)),('edge',np.argsort(edge)[:-16],np.argsort(edge)[-16:])]; folds += [('random'+str(k),np.setdiff1d(np.arange(72),j),j) for k,j in enumerate(np.array_split(rng.permutation(72),4))]
for name,i,j in folds:
 w,rates=amplitudes(i); pred=[F[j]@ls(F[i],Z[i]),F[j]@w@temp(rates).T,F[j]@ls(F[i],Z[i])@np.linalg.pinv(temp([.42,1.8]).T)@temp([.42,1.8]).T]; print(name,'y',[(k,round(np.mean(np.abs(f[j]@ls(f[i],y[i])-y[j])),6)) for k,f in fs.items()],'traj (full, two rise, rounded rates):',[(round(np.mean(abs(v-Z[j])),6),round(np.mean(abs(v[:,-1]-Z[j,-1])),6)) for v in pred], 'rates',rates)
