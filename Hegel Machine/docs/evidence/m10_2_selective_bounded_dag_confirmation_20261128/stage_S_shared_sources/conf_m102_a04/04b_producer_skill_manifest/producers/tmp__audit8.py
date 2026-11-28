import sys,pickle,numpy as np,importlib.util
from scipy.optimize import least_squares
from itertools import combinations
sp=importlib.util.spec_from_file_location('c','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; p=s['panel_c']; u=p['u']; v=p['v1']; lin=m.lin; one=m.one
print('study fields',list(s),'training fields',list(tr),'unique telemetry',np.unique(s['telemetry_vec'])); a,b=m._align(s['panel_a']); z=(x-a)/b; zte=(s['test']['x']-a)/b; print('radii train/test max',np.linalg.norm(z,axis=1).max(),np.linalg.norm(zte,axis=1).max())
for name,fun in [('asinh',np.arcsinh),('atan',np.arctan),('tanh',np.tanh),('softsign',lambda w:w/(1+abs(w))),('softsqrt',lambda w:w/np.sqrt(1+w*w)),('softplusodd',lambda w:np.sign(w)*np.log1p(abs(w)))]:
 def ft(q,X):return np.c_[one(X),fun(q[0]*X)]
 def fit(I):
  def funr(q):
   f=ft(q,u[I]);return (f@lin(f,v[I])-v[I]).ravel()
  p=least_squares(funr,[2],bounds=(.1,35));f=ft(p.x,u[I]);return p.x,lin(f,v[I]),np.std(p.fun)
 q,B,err=fit(np.arange(len(u))); cv=[]
 for I in np.array_split(np.random.default_rng(42).permutation(len(u)),4):
  r,C,_=fit(np.setdiff1d(np.arange(len(u)),I));cv.extend((ft(r,u[I])@C-v[I]).ravel())
 print(name,q,err,np.std(cv),'linear/nonlinear proportional error',np.linalg.norm(B[1:5]-B[5:]*(np.sum(B[1:5]*B[5:])/np.sum(B[5:]**2)))/np.linalg.norm(B[1:5]))
 if name=='asinh':print('asinh coefficients',np.round(B,4))
M=m.fitc(p); C=M[0](z); D=m._square(z)@lin(m._square(u),p['v2']); V=np.c_[C,D]; Y=np.c_[y,T]; N=len(Y); II=np.arange(52); JJ=np.arange(52,N); folds=[np.array_split(np.random.default_rng(k).permutation(II),4) for k in [44,4]]
def kernel(W):
 G=W@W.T/W.shape[1]; dd=G.diagonal(); return G,np.maximum(dd[:,None]+dd-2*G,0)
def vf(V):
 parts=[V,V*V,np.tanh(V),np.sin(V),np.arcsinh(V)]
 for mix in (.5,1,2):
  plus=np.column_stack([V[:,i]+mix*V[:,j] for i,j in combinations(range(8),2)]); minus=np.column_stack([V[:,i]-mix*V[:,j] for i,j in combinations(range(8),2)]);
  parts.extend([np.tanh(plus)/3,np.tanh(minus)/3])
 for alpha in (.15,.5,1.):
  parts.append(np.column_stack([V[:,i]/(1+alpha*V[:,j]**2) for i in range(8) for j in range(8)])/3)
 parts.append(np.column_stack([V[:,i]*V[:,j]*V[:,k] for i,j,k in combinations(range(8),3)])/3)
 return np.column_stack(parts)
W=(V-V[II].mean(0))/V[II].std(0); G,d=kernel(W); F=vf(W); H=F@F.T/F.shape[1]; base=(1+.25*G)**2+np.exp(-.06*d)
rows=[]
for name,K in [('baseline',base),('cubic',(1+.06*G)**3),('quadratic',(1+.25*G)**2),('rational+pair+triple',base+H*.01),('rational+pair+triple strong',base+H*.1),('separable+poly',base+.1*np.exp(-.2*(W[:,:,None]-W.T[None,:,:])**2).mean(1))]:
 for al in [3e-6,3e-5,3e-4,3e-3]:
  def err(I,J):return Y[J]-K[np.ix_(J,I)]@np.linalg.solve(K[np.ix_(I,I)]+al*np.eye(len(I)),Y[I]);
  E=[np.abs(np.vstack([err(np.setdiff1d(II,J),J) for J in fs])).mean(0) for fs in folds]; P=abs(err(II,JJ)).mean(0); rows.append((max(max(e[0],e[1:].mean()) for e in E),name,al,*(np.round([e[0],e[1:].mean(),e[-1]],5).tolist() for e in (*E,P))))
print('inner ranked [y,trajectory,terminal] inner1/inner2/public');print(*sorted(rows)[:16],sep='\n')
