from spatial import *
from scipy.linalg import solve
from itertools import product
Y=np.c_[y,z]; t=s['eval_times']; C=L(phi(x,2),Y); print('time',t,'tele',s['telemetry_vec'],'summary',s['train']['summary']); print('linear',C[:5,[0,1,4,8]].T); print('quadratic singular',np.linalg.svd(C[5:],compute_uv=False));
for j in [0,1,4,8]:
 h=np.zeros((4,4)); h[np.triu_indices(4)]=C[5:,j]; print(j,np.linalg.eigh((h+h.T)/2))
# Test whether a neglected cubic or quartic term is identifiable from training residuals.
J0=np.arange(56,72); folds=[J0,np.arange(16),np.arange(28,44),np.argsort(np.linalg.norm(x,axis=1))[-16:]]
w=coords(x); X=phi(x,2); 
for name,q in w.items():
 P=phi(q,4)[:,15:]; R=Y-X@L(X,Y); P=P-X@L(X,P); P/=np.maximum(P.std(0),1e-8); corr=P.T@R/len(x); inds=np.argsort(np.linalg.norm(corr/R.std(0),axis=1))[-8:]; print(name,'top cub/quart',[(list(cw(range(4),3))[i] if i<20 else list(cw(range(4),4))[i-20],np.round(corr[i,[0,1,4,8]],4)) for i in inds])
# Profile out the linear coefficients of a small number of spatial ridge terms.
def model(q,T,I,r,fn,deg):
 Q=phi(q,deg); scale=np.maximum(T[I].std(0),1e-7); T=T/scale
 Q0=Q[I]; u=np.linalg.qr(Q0)[0]; de=lambda v:v-u@(u.T@v); E=de(T[I]);
 rng=np.random.default_rng(154); starts=[]
 for k in range(18):
  p=rng.normal(size=(r,5)); p[:,0]*=.9; p[:,1:]*=(.55 if k<9 else 1.05); starts.append(p.ravel());
 def extra(q,p):
  p=p.reshape(r,5); a=p[:,0]+q@p[:,1:].T; return fn(a)
 def err(p):
  F=de(extra(q[I],p)); b=L(F,E); return np.r_[(F@b-E).ravel(),.008*p]
 opt=min((least_squares(err,p,bounds=(-6,6),max_nfev=105,ftol=3e-6) for p in starts),key=lambda a:a.cost); M=np.c_[Q,extra(q,opt.x)]; return (M@L(M[I],T[I]))*scale,opt.x
for out in ['y','z','both']:
 for r,deg,f in product([1,2,3],[1,2],['tanh','sin','rat']):
  ee=[]; para=[]
  for J in folds:
   I=np.setdiff1d(np.arange(72),J); _,_,V=np.linalg.svd(z[I],full_matrices=False); V=V[:2]; T={'y':Y[:,:1],'z':z@V.T,'both':np.c_[y,z@V.T]}[out]; pred,p=model(x,T,I,r,{'tanh':np.tanh,'sin':np.sin,'rat':lambda a:1/(1+a*a)}[f],deg); pred=pred if out=='y' else pred@V if out=='z' else np.c_[pred[:,0],pred[:,1:]@V]; G={'y':Y[:,:1],'z':z,'both':Y}[out]; e=abs(pred-G); ee.append(np.r_[e[I].mean(),e[J].mean(),e[J,-1].mean()]); para.append(p)
  print(out,r,deg,f,np.round(np.array(ee).T,5).tolist(),np.round(para[0],2).tolist(),flush=True)
