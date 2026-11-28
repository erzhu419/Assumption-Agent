import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
f=basis(tau); q=fit(tau); jump=h-p[:,-1]; print('ranges',np.c_[q,amp0,dr,jump][:12]); u=np.c_[np.ones(len(h)),dr]; co=np.linalg.lstsq(u,q[:,2],rcond=None)[0];print('drift coeff',co,'resid',q[:,2]-u@co)
from itertools import combinations
z=q-np.c_[amp0,dr]; cols={'1':np.ones(len(h)),'h':h,'j':jump,'j2':jump**2,'j3':jump**3,'absj':abs(jump),'dr':dr,'dr2':dr**2,'h*j':h*jump,'a0':amp0[:,0],'a1':amp0[:,1]};cols.update({f'p{k}':p[:,k] for k in range(p.shape[1])});cols.update({f'x{k}':Xd[:,k] for k in range(4)});cols.update({f'pd{k}':p[:,k+1]-p[:,k] for k in range(p.shape[1]-1)});cols.update({'endavg':p[:,-2:].mean(1),'avg':p@dt/dt.sum(),'jdr':jump*dr,'hdr':h*dr,'z0':h-amp0[:,0]/theta[0],'z1':h-amp0[:,1]/theta[1]})
for j in range(3):
 print('branch',j)
 for k in (1,2,3,4):
  best=[]
  for ks in combinations(list(cols)[1:],k):
   a=np.column_stack([cols[nm] for nm in ('1',)+ks]);Q,R=np.linalg.qr(a);den=1-np.sum(Q**2,1);
   if np.linalg.cond(R)>1e5: continue
   w=np.linalg.solve(R,Q.T@z[:,j]);e=(z[:,j]-a@w)/den;best.append((np.mean(e**2)**.5,ks,w,np.max(abs(e))))
  for row in sorted(best,key=lambda x:x[0])[:3]:print(row)
