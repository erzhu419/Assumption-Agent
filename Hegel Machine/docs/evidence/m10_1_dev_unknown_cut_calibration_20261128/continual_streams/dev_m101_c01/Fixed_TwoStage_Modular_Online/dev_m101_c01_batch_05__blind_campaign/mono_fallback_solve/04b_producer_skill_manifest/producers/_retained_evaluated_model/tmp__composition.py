from spatial import *
from scipy.linalg import solve
from itertools import product
u=s['panel_c']['u']; c=s['panel_c']; tele=s['telemetry_vec']
F={'tanh':np.tanh,'sin':np.sin,'soft':lambda v:v/np.sqrt(1+v*v),'rat':lambda v:v/(1+v*v),'cos':np.cos,'atan':np.arctan}
def learn(k,fn,bias,base):
 v=c[k]; m=F[fn]
 def mat(w,p):
  q=m(w*p[:4]+(p[4:] if bias else 0)); return np.c_[np.ones(len(w)),w,q] if base else np.c_[np.ones(len(w)),q]
 def res(p):
  M=mat(u,p); return (M@L(M,v)-v).ravel()
 init=[1.3] if fn=='cos' else [.6,1.5,2.5,3.5]
 out=min([least_squares(res,np.r_[np.full(4,a),np.full(4,b)] if bias else np.full(4,a),bounds=(np.r_[np.full(4,.05),np.full(4,-1.5)] if bias else .05, np.r_[np.full(4,6),np.full(4,1.5)] if bias else 6),max_nfev=200) for a in init for b in ([0,.4] if bias else [0])],key=lambda r:r.cost)
 M=mat(u,out.x); C=L(M,v); h=np.sum(M*np.linalg.pinv(M).T,1)
 return np.mean(out.fun**2)**.5,np.mean(((M@C-v)/(1-h)[:,None])**2)**.5,out.x,lambda w:mat(w,out.x)@C
best={}
for k in ['v1','v2']:
 rr=[]
 for f,b,base in product(F,[0,1],[0,1]):
  r=learn(k,f,b,base); rr.append((r[0],r[1],f,b,base,r));
 for e,loo,f,b,ba,r in sorted(rr,key=lambda r:r[1])[:8]: print(k,f,b,ba,e,loo,r[2],flush=True)
 best[k]=min(rr,key=lambda r:r[1])[-1][-1]
z0=z; Y=np.c_[y,z]; folds=[(np.arange(56),np.arange(56,72)),(np.arange(16,72),np.arange(16)),(np.r_[0:28,44:72],np.arange(28,44))]; D=[]
for name,w in coords(x).items():
 a,b=[best[k](w) for k in ['v1','v2']];
 for mode,M in {'w':phi(w,2),'c1':phi(a,2),'c2':phi(b,2),'c12':np.c_[phi(a,2),phi(b,2)[:,1:]],'w+c1':np.c_[phi(w,2),a],'w+c2':np.c_[phi(w,2),b],'w+c12':np.c_[phi(w,2),a,b],'w+c2p':np.c_[phi(w,2),phi(b,2)[:,1:]],'w+c12p':np.c_[phi(w,2),phi(a,2)[:,1:],phi(b,2)[:,1:]],'mixed':np.c_[phi(w,2),a,b,w*a,w*b,a*b],'mixedfull':np.c_[phi(w,2),a,b,(w[:,:,None]*a[:,None,:]).reshape(-1,16),(w[:,:,None]*b[:,None,:]).reshape(-1,16),(a[:,:,None]*b[:,None,:]).reshape(-1,16)]}.items():
  for reg in ([0] if M.shape[1]<24 else [])+[.001,.02,.2]:
   err=[]; tr=[]
   for I,J in folds:
    mu=M[I,1:].mean(0); sd=np.maximum(M[I,1:].std(0),.01); Q=np.c_[np.ones(len(M)),(M[:,1:]-mu)/sd]; X=Q[I]; co=solve(X.T@X+reg*np.eye(X.shape[1]),X.T@Y[I],assume_a='pos') if reg else L(X,Y[I]); E=abs(Q@co-Y)
    err.append([E[J,0].mean(),E[J,1:].mean()]);tr.append([E[I,0].mean(),E[I,1:].mean()])
   D.append((name,mode,reg,np.array(err),np.mean(tr,0)))
for k in [0,1]:
 for name,mode,reg,e,tr in sorted(D,key=lambda d:d[3][:,k].mean())[:15]:print(k,name,mode,reg,np.round(e,5).T,'fit',np.round(tr,5))
