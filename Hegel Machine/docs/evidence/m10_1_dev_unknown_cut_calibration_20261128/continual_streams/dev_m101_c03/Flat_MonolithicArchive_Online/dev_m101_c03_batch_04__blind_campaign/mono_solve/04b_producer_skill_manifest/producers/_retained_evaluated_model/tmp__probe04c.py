exec(open('/tmp/probe04.py').read().split("p=S['panel_c']")[0])
from scipy.optimize import minimize_scalar
p=S['panel_c']; u=p['u']; X=calibrate(S,S['train']['x']); t=S['eval_times']; T=S['train']['traj']; y=S['train']['y']; pairs=[(0,1),(0,3),(1,2),(2,3)]
c1=lambda x:np.c_[np.ones(len(x)),*[x[:,i]*x[:,j] for i,j in pairs]]
c2=lambda x,k:np.c_[np.ones(len(x)),np.cos(k*x)]
A=c1(X)@fit(c1(u),p['v1']); k=minimize_scalar(lambda k:np.linalg.norm(c2(u,k)@fit(c2(u,k),p['v2'])-p['v2']),bounds=(1.5,3.1),method='bounded').x; B=c2(X,k)@fit(c2(u,k),p['v2'])
L=np.c_[np.ones(len(X)),A,B]; P=(A[:,:,None]*B[:,None,:]).reshape(-1,16); dot=np.einsum('ii->i',np.eye(4)) if False else P[:,::5].sum(1)
F=lambda r:-np.expm1(-t[:,None]*r)
for train,test in [(np.arange(56),np.arange(56,72)),(np.arange(16,72),np.arange(16))]:
 def sep(r):
  C=fit(F(r),T[train].T); return (F(r)@C-T[train].T).ravel()
 r=least_squares(sep,[.41,1.8]).x; C=fit(F(r),T.T).T
 print('rates',r,'coeff for y, slow, fast (last 16 rows are products)'); print(np.round(fit(np.c_[L,P][train],np.c_[y,C][train]),4))
 # Fit slow-mode residual by adding individual products to a dot-enabled linear baseline.
 H=np.c_[L,dot]; unused=list(range(16)); selected=[]
 for step in range(6):
  if step:
   def loss(j):
    D=np.c_[H,P[:,selected+[j]]];return np.linalg.norm(D[train]@fit(D[train],C[train,0])-C[train,0]);
   j=min(unused,key=loss); unused.remove(j); selected.append(j)
  D=np.c_[H,P[:,selected]]; E=np.c_[H,P[:,selected]] if step>4 else H
  def pred(r,ix):
   f=F(r)[None,:,:]; M=np.concatenate([D[ix,None,:]*f[:,:,0,None],E[ix,None,:]*f[:,:,1,None]],2).reshape(len(ix)*len(t),-1)
   return M
  def loss_r(r):
   M=pred(r,train);return M@fit(M,T[train].ravel())-T[train].ravel()
  opt=least_squares(loss_r,r,bounds=([.03,.6],[1.2,6.])); M=pred(opt.x,train); O=(pred(opt.x,test)@fit(M,T[train].ravel())).reshape(len(test),-1);print(selected,opt.x,np.sqrt(np.mean(opt.fun**2)),np.mean(abs(O-T[test])),np.max(np.mean(abs(O-T[test]),0)))
