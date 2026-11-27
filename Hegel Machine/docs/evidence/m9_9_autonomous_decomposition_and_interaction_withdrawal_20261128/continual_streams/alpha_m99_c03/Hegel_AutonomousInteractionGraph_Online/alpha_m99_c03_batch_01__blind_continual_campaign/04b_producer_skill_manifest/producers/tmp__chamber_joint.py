import numpy as np
z=np.load('/tmp/chamber_probe.npz'); p,h,x,dt,t,tau,th,E,e=[z[k] for k in ('past','hold','features','dt','t','taus','theta','ec','er')]; n=len(h); one=np.ones(n); after=np.cumsum(dt[::-1])[::-1]-dt
m=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-after[:,None]/tau)); r=th[2]+x@th[3:]; F=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]]); Y=np.c_[E,e]
def cv(A,Y):
 B=np.linalg.lstsq(A,Y,rcond=None)[0]; R=Y-A@B; Q=np.linalg.qr(A,mode='reduced')[0]; O=R/(1-(Q**2).sum(1))[:,None]; return B,R,O
for name,A in [('memory',np.c_[one,m]),('memory+drift',np.c_[one,m,r]),('memory+hold+drift',np.c_[one,m,h,r]),('memory+drift features',np.c_[one,m,x]),('memory+hold+drift features',np.c_[one,m,h,x]),('memory+drift interactions',np.c_[one,m,r,h*r,m*r[:,None]]),('past+drift',np.c_[one,p,r])]:
 B,R,O=cv(A,Y); print(name,'rms/max OOF',np.round([np.std(O[:,:-1]),abs(O[:,:-1]).max(),np.std(O[:,-1]),abs(O[:,-1]).max()],5)); print('latent',np.round(B@np.linalg.pinv(F.T),5)); print('end/rate',np.round(B[:,[-2,-1]],5))
Y0=Y-m@(th[:2,None]*F[:,:2].T); print('zero-memory residual versus material drift'); print(np.round(np.c_[r,Y0[:,0],Y0[:,-2:]],5)); print('train drift direction normalized',th[3:]/np.linalg.norm(th[3:]))
for i in [0,1,3,5,8,11,12]:
 co=np.linalg.lstsq(np.c_[one,m,r],Y0[:,i],rcond=None)[0]; print('after subtracting memory',i,np.round(co,5))
