exec(open('/tmp/operator_probe.py').read().split("report('z',z)")[0])
from itertools import combinations,combinations_with_replacement
u,v=s['panel_b']['u'],s['panel_b']['v']; C=np.cos(2.6*z); mix=fit(aff(np.cos(2.6*u)),v); w=aff(C)@mix; y=Y[:,0];
print('mix',mix,'times',t,'w range',w.min(axis=0),w.max(axis=0))
def test(n,a):
 a=aff(a); b=fit(a[:52],y[:52]); e=a@b-y; print(n, '%g %g'%(np.mean(abs(e[:52])),np.mean(abs(e[52:]))),np.round(b,4) if len(b)<20 else '');return e
Q=[]
for n,x in [('w',w),('C',C)]:
 e=test(n,x);q=np.stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)],axis=1);test(n+'quad',np.c_[x,q]);test(n+'cubic',np.c_[x,q,*[np.prod(x[:,ij],axis=1) for ij in combinations_with_replacement(range(4),3)]]); test(n+'pairs',np.c_[x,x[:,0]*x[:,2]+x[:,1]*x[:,3]]);test(n+'signed sum',np.c_[x,x[:,0]*x[:,2]-x[:,1]*x[:,3]]);Q.append(q)
test('w and z',np.c_[w,z]);test('w and z quad',np.c_[w,z,*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]])
base=aff(w); q=Q[0]; names=[f'w{i}w{j}' for i,j in combinations_with_replacement(range(4),2)];
for i,j in [(0,1),(0,2),(0,3)]:
 k,l=sorted(set(range(4))-{i,j});
 for sg in [1,-1]: test(f'paired {i}{j} {sg:+} {k}{l}',np.c_[w,w[:,i]*w[:,j]+sg*w[:,k]*w[:,l]])
for x,n in [(q,'wquad'),(z,'z'),(np.sin(2.15*z),'sinz'),(w*np.roll(w,-1,axis=1),'cycle')]:
 for k in range(1,min(3,x.shape[1])+1):
  def score(jj):
   a=np.c_[base,x[:,jj]]; r=a@fit(a[:52],y[:52])-y;return np.mean(r[:52]**2),np.mean(abs(r[52:])),jj
  print(n,k,min(map(score,combinations(range(x.shape[1]),k))))
print('trajectory residual by time',np.mean(abs(base[52:]@fit(base[:52],tr['traj'][:52])-tr['traj'][52:]),axis=0))
