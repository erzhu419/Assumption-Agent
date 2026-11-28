from finalcv import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations_with_replacement
N=len(x); c=m._coordinates(s,x); g=m._panel_features(s,c)
def dist(z):return (z[:,None,:]-z[None,:,:])**2
def poly(z):return np.column_stack([np.ones(len(z)),z,*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]])
def candidate(coords, features, target, i,j):
 d=dist(coords); f=features@features.T; norm=np.mean(np.diag(f)); f=f/max(norm,1e-8); p=poly(coords); p=p@p.T/np.mean(np.sum(p*p,1)); ii=np.ix_(i,i); ji=np.ix_(j,i); di=d[ii]; fi=f[ii]; pi=p[ii]; y=target[i]; mu=y.mean(0); sig=y.std(0); y=(y-mu)/sig
 def fun(theta):
  a=np.exp(theta); r=np.exp(-di@a[:4]); k=a[4]*r+a[5]*fi+a[6]*pi+a[7]*np.eye(len(i)); cf=cho_factor(k); al=cho_solve(cf,y); inv=cho_solve(cf,np.eye(len(i))); obj=.5*y@al+np.log(np.diag(cf[0])).sum(); w=.5*(inv-np.outer(al,al)); grad=np.r_[-a[:4]*np.einsum('ij,ijk->k',w*a[4]*r,di),np.sum(w*r)*a[4],np.sum(w*fi)*a[5],np.sum(w*pi)*a[6],np.trace(w)*a[7]]; base=np.array([-1.4]*4+[0.,0.,0.,-8.]); reg=np.array([.5]*4+[.05]*3+[.15]); return obj+np.sum(reg*(theta-base)**2),grad+2*reg*(theta-base)
 opt=minimize(fun,[-1.4]*4+[0.,0.,0.,-8.],jac=True,method='L-BFGS-B',bounds=[(-5,1)]*4+[(-9,4)]*3+[(-12,-3)],options={'maxiter':110}); a=np.exp(opt.x); kernel=a[4]*np.exp(-d@a[:4])+a[5]*f+a[6]*p; yp=mu+sig*(kernel[ji]@np.linalg.solve(kernel[ii]+a[7]*np.eye(len(i)),y)); return yp,a
if __name__=='__main__':
 base=run(sp); print('baseline',base[0,:2],base[1:,:2].mean(0),base[1:,:2].max(0)); Z=[]; B=[]
 for i,j in sp:
  pr=m.solve_batch(dict(s,train={**tr,**{k:tr[k][i] for k in ['x','y','traj']}},test={'x':x[j]})); B.append(np.c_[pr['y_hat'],pr['state_pred']]); allp=[]
  for k,l in [(2,0),(2,2),(3,3),(3,1),(0,0),(1,3)]:
   q=np.linalg.svd(Y[i,1:],full_matrices=False)[2][0]; yp,a=candidate(c[k],g[l],Y[:,0],i,j); p1,b=candidate(c[k],g[l],Y[:,1:]@q,i,j); R=Y[i,1:]-np.outer(Y[i,1:]@q,q); qp=np.c_[np.ones(len(x)),c[1],g[3][:,:4]]; residual=qp[j]@np.linalg.solve(qp[i].T@qp[i]+.4*np.eye(qp.shape[1]),qp[i].T@R); allp.append(np.c_[yp,np.outer(p1,q)+residual]);
   if len(Z)==0: print(k,l,'parameters',a,b)
  Z.append(allp)
 Z=np.array(Z); B=np.array(B); truth=np.array([Y[j] for i,j in sp]);
 for name,p in [(str(i),Z[:,i]) for i in range(Z.shape[1])]+[('average',Z[:,:4].mean(1)),('blend',.5*B+.5*Z[:,:4].mean(1))]:
  e=np.abs(p-truth); e=np.c_[e[:,:,0].mean(1),e[:,:,1:].mean((1,2)),e[:,:,-1].mean(1)]; print(name,e[0],e[1:].mean(0),e[1:].max(0))
