exec(open('/tmp/probe98.py').read().split('z,w=m._fit')[0])
z,w=m._fit(m._design(tr,xt,stress,dt,t),*get(tr)); y,r=get(ck); Y=np.c_[y,r]; H=stress(ck['hold_channel_raw']); P=stress(ck['past_channel_raw']); X=xc; F=np.c_[np.ones(len(H)),X]; R=F@w[2:]; decay=-np.expm1(-dt[:,None]/z)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/z); G=H[:,None]-P@decay
B=np.vstack([-np.expm1(-t[:,None]/z),1/z]); A=np.linalg.lstsq(B,Y.T,rcond=None)[0].T
np.set_printoptions(precision=5,suppress=True,linewidth=170)
print('no drift RMSE',np.sqrt(np.mean((A@B.T-Y)**2,axis=0))); print('sweeps',np.c_[*[cal['single_channel_sweeps'][k] for k in ['lab_id','cmd_stress','true_stress']]])
print('H P lab drift A',np.c_[H,P,ck['lab_id'],R,A]); print('feature range diagnostics',X.min(0),X.max(0),'test',xe.min(0),xe.max(0))
from itertools import combinations
cols={'H':H,'R':R,'one':np.ones(len(H)),'Hg0':G[:,0],'Hg1':G[:,1],'Pmean':P.mean(1),'Psum_dt':P@dt,'P0mem':P@decay[:,0],'P1mem':P@decay[:,1]}; cols.update({f'p{i}':P[:,i] for i in range(6)});cols.update({f'x{i}':X[:,i] for i in range(4)})
for b in range(2):
 for n in [1,2,3,4]:
  scored=[]
  for keys in combinations(cols,n):
   Q=np.column_stack([cols[k] for k in keys]); u,*_=np.linalg.lstsq(Q,A[:,b],rcond=None); res=Q@u-A[:,b]; lev=(Q*np.linalg.pinv(Q).T).sum(1); scored.append((np.sqrt(np.mean((res/np.maximum(1-lev,1e-8))**2)),keys,u))
  print('branch',b,'n',n, sorted(scored,key=lambda q:q[0])[:3])
for i,q in enumerate([np.c_[H,P,R],np.c_[H,P,F],np.c_[H,G,R],np.c_[H,R],np.c_[H,R,np.abs(R)],np.c_[H,G,R,R**2]]):
 u=np.linalg.lstsq(q,A,rcond=None)[0]; print('model',i,'coef',u.T,'fit',np.sqrt(np.mean((q@u-A)**2,axis=0)))
