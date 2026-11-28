exec(open('/tmp/checkfinal.py').read().split("p,r,w,e=")[0])
folds=[np.arange(56,72),*np.array_split(np.random.default_rng(123).permutation(n),5),np.argsort(np.linalg.norm(z,axis=1))[-14:]]
for name,slow,cubic in [('full',np.c_[d,c,cross],p3),('c0*d',np.c_[d,c,c[:,0:1]*d],p3),('c0*d1',np.c_[d,c,c[:,0]*d[:,1]],p3),('c0*d1+z',np.c_[d,c,c[:,0]*d[:,1],z],p3),('full+z',np.c_[d,c,cross,z],p3),('c0*d1, cubic+z',np.c_[d,c,c[:,0]*d[:,1]],np.c_[p3,z]),('c0*d, cubic+z',np.c_[d,c,c[:,0:1]*d],np.c_[p3,z])]:
 scores=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); ix=np.r_[tr,te]
  pred,r,_,ir=m._temporal(t,h[tr],[np.c_[c,dot][ix],cubic[ix],slow[ix]],inspect=True); e=abs(pred-h[te]); scores.append([e.mean(),e[:,-1].mean(),ir])
 print(name,'first',np.round(scores[0],5),'mean',np.round(np.mean(scores,0),5),'worst',np.round(np.max(scores,0),5),'tail',np.round(scores[-1],5),'rates',np.round(r,5))
