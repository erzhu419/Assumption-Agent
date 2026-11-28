exec(open('/tmp/dag6.py').read().split('models=[]')[0])
from scipy.special import expit
from scipy.linalg import qr
f2=lambda z,k:np.c_[aff(z),np.sqrt(1+(k*z)**2)-1]
k=least_squares(lambda p:(f2(u,p[0])@ls(f2(u,p[0]),w)-w).ravel(),[1]).x[0];D=lambda z:f2(z,k)@ls(f2(u,k),w)
B=lambda z:aff(z)@b; I=lambda z:(z-b[0])@np.linalg.inv(b[1:]); x=s['train']['x']; z=(x-a[:,0])/a[:,1]; tt=s['eval_times']; T=s['train']['traj']; y=s['train']['y']; print('time',tt,'singular',np.linalg.svd(T,compute_uv=False))
F=lambda p: -np.expm1(-tt[:,None]*p); p=least_squares(lambda p:(F(p)@ls(F(p),T.T)-T.T).ravel(),[1.8,.4]).x; q=ls(F(p),T.T).T; Y=np.c_[y,q]; print('rates',p)
G={'D':D(z),'DB':B(D(z)),'DI':I(D(z)),'BD':D(B(z)),'ID':D(I(z)),'a':z}
folds=np.array_split(np.random.default_rng(202).permutation(len(x)),6)
def library(z):
 P={''.join(map(str,i)):z[:,i].prod(1) for d in range(2,5) for i in it.combinations_with_replacement(range(4),d)}
 Q={str(i):z[:,i] for i in range(4)}
 for i,j in it.combinations(range(4),2):
  for c in (-2,-1,-.5,.5,1,2):Q[f'{i}+{c}*{j}']=z[:,i]+c*z[:,j]
 R=dict(P)
 for key,h in Q.items():
  for c in (.4,1,2,3,5):
   for f,fun in [('th',np.tanh),('sn',np.sin),('cs',np.cos),('rt',lambda h:h/(1+h*h)),('re',lambda h:1/(1+h*h)),('sf',lambda h:h/np.sqrt(1+h*h)),('sq',lambda h:np.sqrt(1+h*h)),('sg',lambda h:np.sign(h)*np.sqrt(np.abs(h)))]:
    if f=='sg' and c!=1:continue
    R[f'{f}{c}({key})']=fun(c*h)
  if len(key)>1 and ('+1*' not in key and '+-1*' not in key):continue
  for nk,v in [*list(Q.items())[:4],*[(key,v) for key,v in P.items() if len(key)==2]]:
   for c in (.5,1,2):R[f'{nk}/({key},{c})']=v/(1+c*h*h)
 return list(R),np.column_stack(list(R.values()))
def greedy(A,H,Y,tr,hi):
 Q=np.linalg.qr(A[tr],mode='reduced')[0]; M=H[tr]-Q@(Q.T@H[tr]); R=Y[tr]-Q@(Q.T@Y[tr]); sel=[];out=[]
 for _ in range(12):
  norm=np.linalg.norm(M,axis=0); idx=np.argmax(np.abs(M.T@R)/np.maximum(norm,1e-12));sel.append(idx);v=M[:,idx]/norm[idx]; M-=np.outer(v,v@M); R-=v*(v@R)
  F=np.c_[A,H[:,sel]];out.append(F[hi]@ls(F[tr],Y[tr]))
 return np.asarray(out),sel
for name,g in G.items():
 keys,H=library(g);A=aff(g); err=np.zeros((12,3));print(name,'coord',g.min(0).round(2),g.max(0).round(2),'linear RMS',np.std(Y-A@ls(A,Y),0))
 for j in range(3):
  for hi in folds:
   tr=np.setdiff1d(np.arange(len(x)),hi);out,_=greedy(A,H,Y[:,j],tr,hi);err[:,j]+=np.abs(out-Y[hi,j]).mean(1)/len(folds)
  _,sel=greedy(A,H,Y[:,j],np.arange(len(x)),[]); print(j,[(i+1,round(err[i,j],4)) for i in (0,1,2,3,5,7,11)], [keys[i] for i in sel[:7]])
