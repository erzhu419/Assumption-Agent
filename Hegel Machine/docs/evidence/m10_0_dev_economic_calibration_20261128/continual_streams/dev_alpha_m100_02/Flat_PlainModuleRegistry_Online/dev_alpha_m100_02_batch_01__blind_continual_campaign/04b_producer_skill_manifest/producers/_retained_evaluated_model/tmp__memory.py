exec(open('/tmp/chamber.py').read().split("print('chamber drift")[0])
from itertools import combinations
lab=c['lab_id']; drift=f@w[2:]; n=len(lab); past,hold,_=s
D=np.c_[np.ones(n),hold,past,drift]; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; tk=lambda u,dd,tt: np.exp(-tt[:,None]/u)*-np.expm1(-dd[:,None]/u)
def score(M):
 E=np.zeros_like(Y); XA=(M[:,:,None,:]*T.T[None,:,:,None]).sum(axis=1)
 for l in np.unique(lab):
  k=lab!=l; q=np.linalg.lstsq(XA[k].reshape(-1,M.shape[-1]),Y[k].ravel(),rcond=None)[0]; E[~k]=(XA[~k]@q-Y[~k]);
 q=np.linalg.lstsq(XA.reshape(-1,M.shape[-1]),Y.ravel(),rcond=None)[0]; return np.round([np.std((XA@q-Y)[:,:-1]),np.std((XA@q-Y)[:,-1]),np.max([np.sqrt(np.mean(E[lab==l,:-1]**2)) for l in np.unique(lab)]),np.max([np.sqrt(np.mean(E[lab==l,-1]**2)) for l in np.unique(lab)])],5), np.round(q,5)
def block(*args):
 out=np.zeros((n,3,sum(x.shape[1] for x in args))); p=0
 for j,x in enumerate(args): out[:,j,p:p+x.shape[1]]=x; p+=x.shape[1]
 return out
for label, x in [('direct',D[:,:-1]),('current',np.c_[D[:,:2],past@tk(tau,dt,tail)]),('reverse',np.c_[D[:,:2],past[:,::-1]@tk(tau,dt,tail)]),('reversed pairs',np.c_[D[:,:2],past[:,::-1]@tk(tau,dt[::-1],tail+dt-np.sum(dt)+np.cumsum(dt)-dt) if False else past@(tk(tau,dt[::-1],np.r_[np.cumsum(dt[::-1][:0:-1])[::-1],0.])[::-1])]),('both',np.c_[D[:,:2],past@tk(tau,dt,tail),past[:,::-1]@tk(tau,dt,tail)])]:
 print(label,score(block(x,x,D[:,[0,-1]])))
print('direct separate coeffs',np.round(np.linalg.lstsq(D,A,rcond=None)[0],5)); print('drift fit and err',np.linalg.lstsq(D[:,[0,-1]],A[:,2],rcond=None)[0],np.std(D[:,[0,-1]]@np.linalg.lstsq(D[:,[0,-1]],A[:,2],rcond=None)[0]-A[:,2])); print('sweep slopes'); sw=cal['single_channel_sweeps']; print([np.polyfit(sw['cmd_stress'][sw['lab_id']==l],sw['true_stress'][sw['lab_id']==l],1) for l in range(4)])
