exec(open('/tmp/invert.py').read().split('for d in range(1,5):')[0])
from itertools import combinations as cb
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=140)
xx=np.r_[s['train']['x'],s['test']['x']]; z=cal(s['panel_a'],xx)[:len(Y)]
def diag(u,d):return np.column_stack([np.ones(len(u))]+[u**k for k in range(1,d+1)])
def solve(A,V):return np.linalg.lstsq(A,V,rcond=None)[0]
def loo(A,V):
 Q=np.linalg.qr(A,mode='reduced')[0]; R=(V-A@solve(A,V))/np.maximum(.003,1-(Q*Q).sum(1))[:,None]; return np.mean(abs(R))
features={'z':z,'x':xx[:len(Y)]}
for key,sub in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 U,V=s[key]['u'],s[key][sub]; print('\n',key,sub,len(U)); W=np.roll(U,-1,axis=1); Z=np.roll(z,-1,axis=1)
 Ds=[('lin',diag(U,1),diag(z,1)),('diag2',diag(U,2),diag(z,2)),('diag3',diag(U,3),diag(z,3)),('diag4',diag(U,4),diag(z,4)),('cyclic',np.c_[diag(U,2),U*W],np.c_[diag(z,2),z*Z]),('quadratic',poly(U,2),poly(z,2))]
 for f in (.5,1,1.25,1.5,2,2.5,3,4):
  Ds += [(f'sin {f}',np.c_[diag(U,2),np.sin(f*U)],np.c_[diag(z,2),np.sin(f*z)]),(f'tanh {f}',np.c_[diag(U,2),np.tanh(f*U)],np.c_[diag(z,2),np.tanh(f*z)]),(f'sincos {f}',np.c_[np.ones(len(U)),np.sin(f*U),np.cos(f*U)],np.c_[np.ones(len(z)),np.sin(f*z),np.cos(f*z)])]
 out=[]
 for name,A,T in Ds:
  for inv in ('direct','atanh'):
   if inv=='atanh' and np.max(abs(V))>=1: continue
   v=V if inv=='direct' else np.arctanh(V); B=solve(A,v); w=T@B; w=w if inv=='direct' else np.tanh(w); mae=loo(A,v); residual=np.mean(abs(A@B-v));
   for d in [1,2,3]:
    C=poly(w,d); pred=C@solve(C[:N],Y[:N]); er=abs(pred[N:]-Y[N:]); out.append((er[:,0].mean(),er[:,1:].mean(),name,inv,d,mae,residual))
   if name in ('lin','diag2','diag3','cyclic','quadratic'): print(name,inv,'LOO',mae,'fit',residual,'coef', np.round(B,3).tolist() if name in ('diag2','diag3') else '')
 for col in [0,1]: print('best',col,sorted(out,key=lambda a:a[col])[:7]);
 print('best panel',sorted({a[2:4]+a[5:] for a in out},key=lambda x:x[2])[:8]);
 for name,A,T in Ds[:6]:features[key+sub+name]=T@solve(A,V)
# Sparse smooth target basis, held out scores, scaling fixed by train.
from sklearn.linear_model import Lasso
D=np.c_[poly(z,4),np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)]
for j,V in enumerate((Y[:,0],Y[:,-1])):
 for alpha in (.0005,.002,.008,.02):
  scale=np.maximum(.001,D[:N].std(0)); A=D/scale; m=Lasso(alpha=alpha,max_iter=20000).fit(A[:N,1:],V[:N]); p=m.predict(A[N:,1:]); i=np.argsort(abs(m.coef_))[-14:]+1; print('sparse',j,alpha,np.mean(abs(p-V[N:])),list(zip(i,(m.coef_[i-1]/scale[i]).round(3))))
