import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/graph.py').read())
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import KFold
h=lambda u:np.sign(u)*(-np.expm1(-1.8*np.abs(u)))
j=lambda u:np.logaddexp(0,2.2*u)
def op(f,V):
 W=fit(aff(f(U)),V); O,_=orthogonal_procrustes(f(U)-f(U).mean(0),V-V.mean(0)); q=np.sum(((f(U)-f(U).mean(0))@O)*(V-V.mean(0)))/np.sum((f(U)-f(U).mean(0))**2); print('panel errors and scale',np.std(aff(f(U))@W-V),np.std((f(U)-f(U).mean(0))@O*q+V.mean(0)-V),q); return lambda z:(f(z)-f(U).mean(0))@O*q+V.mean(0)
H=op(h,R)(z); J=op(j,S)(z)
print('A calibration',coef,'clean residual',np.std((v-u*coef[:,1]-coef[:,0])[mask],0))
X=np.einsum('ni,nj->nij',H,J).reshape(-1,16); L=np.c_[H,J]; diag=(H*J).sum(1); W=aff(np.c_[L,X]); w=fit(W,Y);print('linear weights',w[:9]);print('cross weights',w[9:].reshape(4,4,3).transpose(2,0,1)); print('resid std',np.std(W@w-Y,0));
for name,E in [('full',W),('diag',aff(np.c_[L,diag])),('individual diag',aff(np.c_[L,H*J])),('H diag',aff(np.c_[H,diag])),('J diag',aff(np.c_[J,diag])),('diag+J2',aff(np.c_[H,poly(J,2)[:,1:],diag])),('diag+H2',aff(np.c_[J,poly(H,2)[:,1:],diag])),('full+J2',aff(np.c_[H,poly(J,2)[:,1:],X])),('full+H2',aff(np.c_[J,poly(H,2)[:,1:],X]))]:
 PP=np.zeros_like(Y);PT=np.zeros_like(T)
 folds=[(np.arange(56),np.arange(56,72))]+list(KFold(6,shuffle=True,random_state=932).split(z)); scores=[]
 for a,c in folds:
  r=least_squares(lambda p:(td(p)@fit(td(p),T[a].T)-T[a].T).ravel(),[.4,1.8]); Dt=td(r.x); ytr=np.c_[fit(Dt,T[a].T).T,y[a]]; P=E[c]@fit(E[a],ytr);PP[c]=P;PT[c]=P[:,:2]@Dt.T; scores.append([np.mean(abs(P[:,2]-y[c])),np.mean(abs(PT[c]-T[c]),0).max()]);
 print(name,'train rms',np.std(E@fit(E,Y)-Y,0),'split',scores[0],'OOF',[np.mean(abs(PP[:,2]-y)),np.mean(abs(PT-T),0).max()],'worst',np.max(scores[1:],0))
