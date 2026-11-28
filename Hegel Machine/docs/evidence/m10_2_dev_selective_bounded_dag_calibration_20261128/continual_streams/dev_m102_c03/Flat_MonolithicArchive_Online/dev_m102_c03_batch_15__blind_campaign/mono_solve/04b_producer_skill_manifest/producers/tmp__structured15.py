exec(open('/tmp/compact15.py').read().split('q=lambda')[0])
from scipy.optimize import least_squares
cross=(c[:,:,None]*d[:,None,:]).reshape(len(x),-1); dot=(c*d).sum(1); one=np.ones(len(x)); q=lambda r:-np.expm1(-t[:,None]*r)
scalar=[('diag',np.c_[one,c,d,c*d]),('tied-dot',np.c_[one,z,c,d,dot]),('tied-both',np.c_[one,z,c.sum(1)+d.sum(1),dot])]
for name,F in scalar:
 e=[abs(F[b]@ls(F[a],y[a])-y[b]).mean() for a,b in splits]; print('scalar',name,e[0],max(e[1:21]),max(e[21:]))
for name,L,R in [('strict',np.c_[one,d,cross[:,1]],np.c_[one,c,dot]),('linear-z',np.c_[one,z,d,cross[:,1]],np.c_[one,z,c,dot]),('all-affine',np.c_[one,z,c,d,cross[:,1]],np.c_[one,z,c,d,dot]),('full',f,f)]:
 E=[]; rates=[]; costs=[]
 for a,b in splits:
  def matrix(ix,r):
   waves=q(r)
   return np.c_[(L[ix,None,:]*waves[None,:,0,None]).reshape(len(ix)*len(t),-1),(R[ix,None,:]*waves[None,:,1,None]).reshape(len(ix)*len(t),-1)]
  def res(r):
   g=matrix(a,r); val=T[a].ravel(); return g@ls(g,val)-val
  r=least_squares(res,[.42,1.81],bounds=([.12,1],[.9,4])).x
  rates.append(r); G=matrix(a,r); B=ls(G,T[a].ravel()); costs.append(np.mean(res(r)**2)**.5); er=abs((matrix(b,r)@B).reshape(len(b),-1)-T[b]); E.append([er.mean(),er.mean(0).max(),er.max()]);
  if len(E)==1: print('coeffs',name,r,np.round(B,5))
 E=np.array(E); print('traj',name,'train RMSE',costs[0], 'official,random,tails',np.round(E[0],5),np.round(E[1:21].max(0),5),np.round(E[21:].max(0),5),'rate spread',np.std(rates,0))
