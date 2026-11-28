exec(open('/tmp/validate_batch04.py').read().split('records=[]')[0])
from scipy.optimize import least_squares
T=np.asarray(s['eval_times']); F=a[:,-1:]; P=a[:,1:5]; Q=a[:,5:9]; C=b[:,10:11]; D=b[:,[9,14,19,24]]; E=b[:,[10,15,20,21]]
options={'compact':([np.c_[P,D],np.c_[Q,E]],[0,1]),'tied':([np.c_[P,F],np.c_[Q,C]],[0,1]),'tied-independent-cross':([P,Q,F,C],[0,1,2,3]),'tied-independent-diagonal':([P,np.c_[Q,C],F],[0,1,2]),'tied-independent-offdiag':([np.c_[P,F],Q,C],[0,1,2])}
for name,(blocks,rs) in options.items():
 def design(i,p):
  return np.concatenate([(-np.expm1(-T*p[r]))[None,:,None]*g[i,None,:] for r,g in zip(rs,blocks)],axis=-1).reshape(len(i)*len(T),-1)
 def fit(i,j):
  obs=t[i].ravel()
  def err(p):
   d=design(i,p);return d@m._ls(d,obs)-obs
  initial=[1.82,.56]+[1.5 if k==2 and len(rs)==4 or name=='tied-independent-diagonal' else .56 for k in range(2,len(rs))]; p=least_squares(err,initial,bounds=([.06]*len(initial),[5.]*len(initial))).x
  d=design(i,p);co=m._ls(d,obs);e=abs((design(j,p)@co).reshape(t[j].shape)-t[j]);return e.mean(),e.mean(0).max(),e.max(),p,co
 rec=[(label,*fit(i,j)[:3]) for label,i,j in splits];print(name,'official',rec[0],'worst',*[max(rec,key=lambda r:r[k]) for k in [1,2,3]]); print('full',fit(np.arange(72),np.arange(72)))
