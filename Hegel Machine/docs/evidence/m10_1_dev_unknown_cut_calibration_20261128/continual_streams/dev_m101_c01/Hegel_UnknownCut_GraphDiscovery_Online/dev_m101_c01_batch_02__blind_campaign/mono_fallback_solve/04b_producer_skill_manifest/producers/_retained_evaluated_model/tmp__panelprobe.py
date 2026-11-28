from structure import *
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from itertools import combinations

def extras(u):
 cols=[]; labels=[]
 def add(label,col): labels.append(label);cols.append(col)
 for j in range(4):
  v=u[:,j]
  for k in (.7,1.2,1.7,2.3):
   for fn in (np.sin,np.cos,np.tanh): add((fn.__name__,j,k),fn(k*v))
  for h in range(4):
   w=u[:,h];add(('rational',j,h),v/(1+w*w));add(('tanhProduct',j,h),v*np.tanh(w)); add(('sinProduct',j,h),v*np.sin(w));add(('denomProduct',j,h),v*w/(1+w*w))
 for j,h in combinations(range(4),2):
  for sign in (-1,1):
   for k in (.75,1.25,1.7):
    for fn in (np.sin,np.cos,np.tanh):add((fn.__name__,j,h,sign,k),fn(k*(u[:,j]+sign*u[:,h])))
 return np.column_stack(cols),labels

def examine_c():
 C=s['panel_c'];u=C['u']; v=np.c_[C['v1'],C['v2']]; base=poly(u,1); extra,labs=extras(u); P=lambda a:a-base@np.linalg.lstsq(base,a,rcond=None)[0];r=P(v);F=P(extra); corr=F.T@r/(np.linalg.norm(F,axis=0)[:,None]*np.linalg.norm(r,axis=0)[None,:]+1e-16)
 for j in range(8):
  ix=np.argsort(abs(corr[:,j]))[-3:][::-1]; print('C channel',j,[(labs[i],round(corr[i,j],4)) for i in ix]);
 for d in (1,2,3):
  f=poly(u,d);Q=np.linalg.qr(f,mode='reduced')[0]; r=v-Q@(Q.T@v);print('C poly',d,'fit',np.sqrt((r*r).mean(0)), 'LOO MAE',np.abs(r/(1-(Q*Q).sum(1)+1e-12)[:,None]).mean(0))
 return base,extra,labs

def poly_kernel(u,weights):
 f=[poly(u,1)];
 for d,w in enumerate(weights,2):
  q=poly(u,d);cols=q[:,poly(u,d-1).shape[1]:];f.append(np.sqrt(w)*cols)
 f=np.column_stack(f);return f@f.T

def comparison():
 from scipy.optimize import minimize
 from scipy.linalg import eigh
 rng=np.random.default_rng(511); ids=np.arange(72); splits=[(ids[:56],ids[56:])]+[(i[:56],i[56:]) for i in [rng.permutation(ids) for _ in range(4)]];splits.append((np.argsort(np.linalg.norm(x,axis=1))[:56],np.argsort(np.linalg.norm(x,axis=1))[56:]))
 # Tune kernel amplitudes and ridge jointly by marginal likelihood, using training rows only.
 def run(name,blocks):
  kernels=np.stack(blocks+[np.eye(len(x))]); kernels=kernels/(np.mean(np.diagonal(kernels,axis1=1,axis2=2),axis=1)[:,None,None]); results=[]; params=[]
  for i,j in splits:
   K=kernels[:,i[:,None],i]; KP=kernels[:,j[:,None],i]; Y=np.c_[y,z][i]; scale=Y.std(0);Y=Y/scale; out=[]; ps=[]
   for U in (Y[:,:1],Y[:,1:]):
    # Shared covariance across trajectory columns, with the same likelihood units as the scalar fit.
    def ml(q):
     w=np.exp(q); c=cho_factor(np.einsum('k,kij->ij',w,K),check_finite=False); a=cho_solve(c,U,check_finite=False); L=cho_solve(c,np.eye(len(i)),check_finite=False); m=U.shape[1]; val=(a*U).sum()/2+m*np.log(np.diag(c[0])).sum(); g=w*np.einsum('kij,ij->k',K,m*L-a@a.T)/2; return val,g
    fit=minimize(ml,np.r_[np.zeros(len(K)-1),-8.],jac=True,bounds=[(-19,8)]*(len(K)-1)+[(-18,-1)],method='L-BFGS-B'); w=np.exp(fit.x); ps.append(w); out.append(np.einsum('k,kij->ij',w,KP)@np.linalg.solve(np.einsum('k,kij->ij',w,K),U))
   err=abs(np.column_stack(out)*scale-np.c_[y,z][j]); results.append([err[:,0].mean(),err[:,1:].mean()]);params.append(ps)
  print(name,np.round(results,5).T,'weights first',np.round(params[0],5))
 for name,u in [('X',x),('A',xa),('B',xb),('AB',ab),('BA',ba)]:
  p=poly(u,2); p3=poly(u,3)[:,15:]; e,_=extras(u); e-=p@np.linalg.lstsq(p,e,rcond=None)[0]; e/=np.maximum(np.linalg.norm(e,axis=0),1e-10); d=cdist(u,u,'sqeuclidean');q=np.sqrt(5*d)/2; run(name+' RBF',[p@p.T,np.exp(-d/8), (1+q+q*q/3)*np.exp(-q)]); run(name+' grammar',[p@p.T,p3@p3.T,e@e.T,np.exp(-d/8)]);
if __name__=='__main__':
 examine_c(); comparison()
