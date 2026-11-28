exec(open('/tmp/structure.py').read().split('Y=[]; T=[]')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
np.set_printoptions(precision=4,suppress=True,linewidth=160)
q=lambda z: np.c_[np.ones(len(z)),z[:,[0,0,1,2]]*z[:,[1,3,2,3]]]
c=q(z)@ls(q(u),p['v1']); d=m._branch(p,z,2)
prod=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(z),-1)
cubidx=np.array(list(combinations_with_replacement(range(4),3)))
cub=np.prod(z[:,cubidx],axis=-1); ZC=prod(z,c); D0=d*c; DC=prod(d,c); ZD=prod(z,d); Bz=ones(z)@B
h=lambda w:-np.expm1(-s['eval_times'][:,None]*w)
project=lambda tr: (lambda H:H@np.linalg.pinv(H))(h(least_squares(lambda w:(h(w)@ls(h(w),tr.T)-tr.T).ravel(),[1.8,.4],bounds=([1,.1],[3,.95])).x))
base=ones(np.c_[c,d]); bc=ones(np.c_[z,c,d]);
families={'zc':ZC,'cubic':cub,'diag':D0,'cross':DC,'zd':ZD,'bc':Bz*c,'cc':prod(c,c),'dd':prod(d,d),'bd':Bz*d}
def loss(F):
 out=[]
 for r,v in cuts:
  pred=F[v]@ls(F[r],np.c_[y,t][r]); out.append([abs(pred[:,0]-y[v]).mean(),abs(pred[:,1:]@project(t[r])-t[v]).mean()])
 return np.max(out,axis=0),np.mean(out,axis=0)
results=[]
for bb,F in [('cd',base),('zcd',bc)]:
 for k in range(4):
  import itertools
  for ks in itertools.combinations(families,k):
   X=np.column_stack([F]+[families[key] for key in ks]);
   if np.linalg.matrix_rank(X)>49: continue
   ww,mm=loss(X); tr=abs(X@ls(X,np.c_[y,t])-np.c_[y,t]).mean(0); results.append((ww,mm,bb,ks,np.linalg.matrix_rank(X),tr[0],tr[1:].mean()))
for j in [0,1]:
 print('RESULTS',j)
 for rr in sorted(results,key=lambda a:a[0][j])[:15]:print(rr)
print('project full MAE',abs(t@project(t)-t).mean())
# Identify term coefficients in the compact basis whose scalar fit is near the measurement floor.
labels=['intercept']+[f'c{i}' for i in range(4)]+[f'd{i}' for i in range(4)]+[f'z{i}c{j}' for i in range(4) for j in range(4)]+[f'c{i}d{i}' for i in range(4)]
F=np.c_[base,ZC,D0]; print('zc diag',loss(F)); print('full zc diag coefficients',list(zip(labels,np.round(ls(F,np.c_[y,t])[:,[0,1,8]],4))))
