exec(open('/tmp/explore_chamber.py').read().split('for tt in [taus]:')[0])
from itertools import combinations
h=cq[1]; past=cq[0]; drift=cq[2]@th[2:]; print('diagnostic h and history ranges',np.round(np.min(np.c_[h,past],axis=0),3),np.round(np.max(np.c_[h,past],axis=0),3))
print('test history ranges',np.round(np.min(np.c_[b[1],b[0]],axis=0),3),np.round(np.max(np.c_[b[1],b[0]],axis=0),3))
sc=np.r_[np.ones(len(t)),.9]
def cb(tt,lin): return basis(tt) if lin else basis(tt)[:,:2]
for lin in (False,True):
 def res(l):
  B=cb(np.exp(l),lin)*sc[:,None]; Y=y*sc; return (np.linalg.lstsq(B,Y.T,rcond=None)[0].T@B.T-Y).ravel()
 f=least_squares(res,np.log(taus),bounds=(np.log(.05),np.log(25))); tt=np.exp(f.x); B=cb(tt,lin); coef=np.linalg.lstsq(B*sc[:,None],(y*sc).T,rcond=None)[0].T
 print('free tau, drift term?',lin,tt,stat(coef@B.T-y)); print('coef, effective fast memory, slow minus theta1 hold, drift'); print(np.round(np.c_[coef,h-coef[:,0]/th[0],coef[:,1]-h*th[1],drift],4))
z=(m._design(cq,dt,np.array([1.]),taus)[:,-1,:2]*taus); z=h[:,None]-z
A={'1':np.ones(len(h)),'h':h,'h2':h*h,'h3':h**3,'z0':z[:,0],'z1':z[:,1],'dr':drift,'absdr':abs(drift),'dr+':np.maximum(drift,0),'hdr':h*drift,'dr2':drift**2,'mean':past.mean(1),'rev0':h-m._design((past[:,::-1],h,cq[2]),dt,np.array([1.]),taus)[:,-1,0]*taus[0]}
for j in range(6): A['p'+str(j)]=past[:,j]
for j in range(4): A['x'+str(j)]=xq[:,j]
B=cb(np.array([.475,2.69]),False); coef=np.linalg.lstsq(B*sc[:,None],(y*sc).T,rcond=None)[0].T
for j in range(2):
 print('branch',j)
 for size in (1,2,3):
  rows=[]
  for keys in combinations(A,size):
   Z=np.column_stack([A[k] for k in keys]); Q,R=np.linalg.qr(Z); fit=np.linalg.lstsq(Z,coef[:,j],rcond=None)[0]; e=(Z@fit-coef[:,j]); score=np.sqrt(np.mean((e/np.maximum(1-np.sum(Q*Q,axis=1),.0001))**2)); rows.append((score,keys,np.round(fit,3)))
  print(sorted(rows,key=lambda x:x[0])[:4])
