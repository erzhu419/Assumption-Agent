exec(open('/tmp/readout_diag.py').read().split('for keys in')[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
C=s['panel_c'];u=C['u']; Q=np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]); P=m._cycle(u); print('C1 basis residuals',[(D.shape[1],np.sqrt(np.mean((D@m._fit(D,C['v1'])-C['v1'])**2))) for D in [m._affine(P),m._affine(u),np.c_[m._affine(u),P],np.c_[m._affine(u),Q]]]);print('A matrix',m._fit(m._affine(s['panel_a']['u']),s['panel_a']['v'])); print('times',s['eval_times']);
_,_,V=np.linalg.svd(tr,full_matrices=False); T=np.c_[y,tr@V[:2].T]; folds += [(np.argsort(z[:,j])[:-10],np.argsort(z[:,j])[-10:]) for j in range(4)]
sets={'z':z,'x':x,'b':b,'rb':rb,'d':c2,'c':c1,'zb':np.c_[z,b], 'xb':np.c_[x,rb]}
def scores(pred,iv):return [np.mean(abs(pred[:,0]-y[iv])),np.mean(abs(pred[:,1:]@V[:2]-tr[iv]))]
for name,A in sets.items():
 A=(A-A.mean(0))/A.std(0); ds=(A[:,None,:]-A[None,:,:])**2; nfeat=A.shape[1]; I=np.eye(len(A)); af=m._affine(A); lin=af@af.T/nfeat
 for kind in ['radial','add','mix','product']:
  out=[]; pars=[]
  for it,iv in folds:
   Y=T[it]; center=Y.mean(0); scale=Y.std(0); Y=(Y-center)/scale; p0=np.r_[np.full(nfeat,-1.7),-8.,-2.]; idx=np.ix_(it,it)
   def kern(p,grad=False):
    w=np.exp(p[:nfeat]); d=ds*w; E=np.exp(-d); r=np.exp(-d.sum(-1)); a=E.mean(-1); v=np.exp(p[-1]); noise=np.exp(p[-2]);
    if kind=='radial': K=r; G=-d*r[:,:,None]
    if kind=='add': K=a; G=-d*E/nfeat
    if kind=='mix': K=.5*(r+a); G=-.5*d*(r[:,:,None]+E/nfeat)
    if kind=='product': K=a*a; G=-2*d*E*a[:,:,None]/nfeat
    return (K+v*lin+noise*I,np.concatenate((G,(noise*I)[:,:,None],(v*lin)[:,:,None]),axis=2)) if grad else K+v*lin+noise*I
   def loss(p):
    K,G=kern(p,True); f=cho_factor(K[idx]); al=cho_solve(f,Y); inv=cho_solve(f,np.eye(len(it))); W=Y.shape[1]*inv-al@al.T; reg=.1*(p-p0); return .5*np.sum(al*Y)+Y.shape[1]*np.log(f[0].diagonal()).sum()+.05*np.sum((p-p0)**2), .5*np.einsum('ij,ijk->k',W,G[idx])+reg
   fit=minimize(loss,p0,jac=True,method='L-BFGS-B',bounds=[(-8,3)]*nfeat+[(-13,-2),(-8,3)],options={'maxiter':75}); K=kern(fit.x); P=K[np.ix_(iv,it)]@cho_solve(cho_factor(K[idx]),Y)*scale+center;out.append(scores(P,iv)); pars.append(fit.x)
  print(name,kind,'official',np.round(out[0],4),'random max',np.round(np.max(out[:5],0),4),'boundary max',np.round(np.max(out[5:],0),4),'p',np.round(pars[0],2),flush=True)
