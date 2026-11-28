from spatial import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import product

def main():
 u=s['panel_c']['u']; ones=lambda w:np.ones((len(w),1)); U=np.c_[ones(u),u]
 P={} 
 for k in ('v1','v2'):
  v=s['panel_c'][k]; results=[]
  forms={'atan':(1,[2.35],[(.2,5)],lambda w,p:np.arctan(p[0]*w)), 'tanh':(1,[2.2],[(.2,5)],lambda w,p:np.tanh(p[0]*w)), 'cos':(1,[1.28],[(.4,3)],lambda w,p:np.cos(p[0]*w)), 'tanh+sin':(3,[1.7,.2,3.],[(.2,5),(-2,2),(.7,6)],lambda w,p:np.tanh(p[0]*w)+p[1]*np.sin(p[2]*w)), 'atan+tanh':(3,[2.35,.2,1.7],[(.2,5),(-2,2),(.2,5)],lambda w,p:np.arctan(p[0]*w)+p[1]*np.tanh(p[2]*w)), 'cos+cos':(3,[1.3,.05,2.6],[(.4,2),(-1,1),(2,5)],lambda w,p:np.cos(p[0]*w)+p[1]*np.cos(p[2]*w)), 'tanh+cube':(2,[1.8,.1],[(.2,5),(-1,1)],lambda w,p:np.tanh(p[0]*w)+p[1]*w**3)}
  for name,(_,p,bd,fn) in forms.items():
   def mat(w,p):return np.c_[ones(w),w,fn(w,p)]
   def err(p):
    X=mat(u,p); Q=np.linalg.pinv(X); return ((X@(Q@v)-v)/(1-np.sum(X*Q.T,1))[:,None]).ravel()
   opt=least_squares(err,p,bounds=np.array(bd).T,max_nfev=150); m=mat(u,opt.x); c=L(m,v)
   results.append((np.std(opt.fun),name,opt.x,fn,c)); print('panel',k,name,np.std(opt.fun),opt.x,flush=True)
  _,name,p,fn,c=min(results,key=lambda r:r[0]); print('choice',k,name,np.round(c,3)); P[k]=lambda w,fn=fn,p=p,c=c:np.c_[ones(w),w,fn(w,p)]@c
 Y=np.c_[y,z]; qq=phi(x,2); R=Y-qq@L(qq,Y); print('raw residual SVD',np.linalg.svd(R,compute_uv=False)); print('residual corr',np.corrcoef(R.T)[0]);print('y start/end corr',np.corrcoef(Y.T)[0]); print('y from true traj diagnostic',np.mean(abs(y-phi(np.c_[z[:,0],z[:,-1],z[:,3],z[:,4]],2)@L(phi(np.c_[z[:,0],z[:,-1],z[:,3],z[:,4]],2),y))))
 folds=[np.arange(56,72),np.arange(16),np.arange(28,44),np.argsort(np.linalg.norm(x,axis=1))[-16:]]
 def norm(q):
  q=q-q.mean(0); return q/np.maximum(q.std(0),1e-6)
 def kern(w):
  g=norm(w); r=g@g.T/w.shape[1]; return [r,r**2,r**3,r**4]
 def assess(groups):
  W=np.array([np.ones((len(x),len(x)))]+groups); n=len(W); results=[]
  for J in folds:
   I=np.setdiff1d(np.arange(len(x)),J); trs=[]
   for T in [Y[:,:1],Y[:,1:]]:
    T=T[I]; mu=T.mean(0); sd=T.std(0); T=(T-mu)/sd
    if T.shape[1]>1:
     _,_,V=np.linalg.svd(T,full_matrices=False); V=V[:2]; T=T@V.T; rs=T.std(0); T=T/rs
    C=np.concatenate((W[:,I][:,:,I],np.eye(len(I))[None]),axis=0); E=np.eye(len(I)); m=T.shape[1]
    def loss(p):
     a=np.exp(p); K=np.einsum('i,ijk->jk',a,C); F=cho_factor(K,lower=True); alpha=cho_solve(F,T); invers=cho_solve(F,E); Jk=m*invers-alpha@alpha.T; return .5*np.sum(T*alpha)+m*np.log(F[0].diagonal()).sum(),.5*a*np.einsum('jk,ijk->i',Jk,C)
    opt=minimize(loss,np.r_[0,np.full(n-1,-3),-8],jac=True,method='L-BFGS-B',bounds=[(-16,5)]*n+[(-16,0)],options={'maxiter':160}); a=np.exp(opt.x); pred=np.einsum('i,ijk->jk',a[:-1],W[:,J][:,:,I])@cho_solve(cho_factor(np.einsum('i,ijk->jk',a,C),lower=True),T)
    if T.shape[1]>1: pred=(pred*rs)@V
    trs.append(pred*sd+mu)
   e=np.abs(np.c_[*trs]-Y[J]); results.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
  return np.array(results).round(5).T
 cur=coords(x); cur['abi']=(cur['a']-B[0])@np.linalg.inv(B[1:]); cur['bia']=(cur['bi']-A[0])/A[1:].diagonal()
 groups=kern(x); print('K raw',assess(groups),flush=True)
 for name,w in cur.items():
  a,b=P['v1'](w),P['v2'](w); g=norm(w); sr=np.exp(-.5*(g[:,None,:]-g[None,:,:])**2)
  add=[sr[:,:,i] for i in range(4)]; pair=[(sr[:,:,i]*sr[:,:,j]) for i in range(4) for j in range(i+1,4)]
  for flavor,ww in [('add',groups+add+pair),('panel',groups+kern(a)[:3]+kern(b)[:3]),('interaction',groups+kern(a)[:2]+kern(b)[:2]+[g@norm(g*norm(b)).T if False else kern(np.c_[w,a,b])[j] for j in (1,2)]),('trig',groups+[norm(f(w))@norm(f(w)).T/4 for f in (lambda q:np.arctan(2.3*q),lambda q:np.cos(1.28*q),lambda q:np.sin(1.7*q))]+add)]:
   print('K',name,flavor,assess(ww),flush=True)
if __name__=='__main__':main()
