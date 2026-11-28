from operators import *
from itertools import combinations, product
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
x=s['train']['x']; z=(x-a)/b; n=52; t=s['eval_times']; tr=s['train']['traj']; H=-np.expm1(-t[:,None]*np.array([.38,1.8,5.4])); Y=np.c_[s['train']['y'],fit(H,tr.T).T]; Q=PolynomialFeatures(2).fit_transform(z); P=PolynomialFeatures(3).fit_transform(z); Z={'z':z,'B':Bf(z),'Bt':(z-B[0])@B[1:].T,'BL':z@B[1:],'BT':z@B[1:].T}; c=s['panel_c']; u=c['u']; C=fit(PolynomialFeatures(2).fit_transform(u),c['v1']); C[abs(C)<.0005]=0; G=fit(F(np.cos(2.4*u)),c['v2']);
for k,w in list(Z.items()):Z[k+'q']=PolynomialFeatures(2).fit_transform(w)@C; Z[k+'c']=F(np.cos(2.4*w))@G
scores=[]; preds=[]
def ridge(K,d,l,key):
 D=d[:n]; N=np.column_stack((K[:n,:n]+l*np.eye(n),D)); U=np.r_[N,np.c_[D.T,np.zeros((D.shape[1],D.shape[1]))]]; I=np.linalg.inv(U)[:n,:n]; A=I@Y[:n]; beta=fit(D,Y[:n]-N[:,:n]@A); pred=K[n:,:n]@A+d[n:]@beta; e=A/np.diag(I)[:,None]; scores.append((abs(e).mean(0),abs(pred-Y[n:]).mean(0),key));preds.append(pred)
def kk(M):
 M=M-M[:n].mean(0); M=M/np.maximum(M[:n].std(0),.03);return M@M.T/max(1,M.shape[1])
blocks={}; Db={}
for k,w in Z.items():
 if len(k)>2 and k not in ['BL','BT']: continue
 v1=np.column_stack([w[:,i]+sg*w[:,j] for i,j in combinations(range(4),2) for sg in [1,-1]])
 v2=np.column_stack([w[:,i]*w[:,j] for i,j in combinations(range(4),2)])
 v3=np.column_stack([w[:,i]**2+sg*w[:,j]**2 for i,j in combinations(range(4),2) for sg in [1,-1]])
 v4=np.c_[v1[:,:4]*w[:,2,None],v1[:,4:8]*w[:,3,None],w[:,0]*w[:,2]*w[:,3],w[:,0]*w[:,1]*w[:,2],w[:,1]*w[:,2]*w[:,3],w[:,0]*w[:,1]*w[:,3]]
 Db[k]={'lin':np.c_[w,v1],'product':v2,'sqpair':v3,'triple':v4}
 for nm,V in Db[k].items():
  for power in [.625,1.25,2.4,3.75]:
   for f,fun in [('trig',lambda d:np.c_[np.sin(d),np.cos(d)]),('abs',np.abs),('rational',lambda d:np.c_[d/(1+d*d),1/(1+d*d)]),('sat',lambda d:np.c_[np.tanh(d),d/(1+abs(d))])]:
    T=fun(V*power); blocks[k,nm,power,f]=kk(T)
   if nm=='lin':
    T=np.cos(power*V); blocks[k,nm,power,'lintrig']=kk((w[:,:,None]*T[:,None,:]).reshape(len(w),-1))
for key,K in list(blocks.items()):
 for l in [.0002,.005,.05,.2]:ridge(K+.2*kk(P[:,15:]),Q,l,key+(l,))
for k,w in Z.items():
 d=cdist(w,w,'sqeuclidean');
 for g in [.02,.07,.2,.55,1.2,2.5]:
  for l in [.0001,.001,.01,.05,.2]:
   for mode,D in enumerate([F(z),Q]):
    for name,K in [('rbf',np.exp(-g*d)),('imq',1/np.sqrt(1+g*d)),('matern',(1+np.sqrt(g*d))*np.exp(-np.sqrt(g*d)))]:ridge(K,D,l,(k,name,g,l,mode))
# Sparse one-feature augmentation, residualized on quadratic backbone. Rank with leverage-corrected training predictions.
for k, vv in Db.items():
 for nm,v in vv.items():
  for i,d in enumerate(v.T):
   for gain in [.22,.625,1.25,2.4,3.75]:
    for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('abs',np.abs),('rat',lambda w:1/(1+w*w))]:
     j=f(gain*d)
     for mm,D in [('add',np.c_[Q,j]),('lin',np.c_[Q,F(z)*j[:,None]]),('uv',np.c_[Q,v*j[:,None]])]:
      U,S,V=np.linalg.svd(D[:n],full_matrices=False); use=S>S[0]*1e-6; U=U[:,use]; B=(V[use].T/S[use])@(U.T@Y[:n]); p=D[n:]@B; e=(Y[:n]-D[:n]@B)/(1-(U*U).sum(1)[:,None]); scores.append((abs(e).mean(0),abs(p-Y[n:]).mean(0),(k,nm,i,gain,fn,mm)));preds.append(p)
S=np.array([v[0] for v in scores]); E=np.array([v[1] for v in scores]);
for j in range(4):
 print('OUTPUT',j)
 for order in [np.argsort(S[:,j]),np.argsort(E[:,j])]:
  for i in order[:5]:print(np.round(S[i,j],4),np.round(E[i,j],4),scores[i][2])
for k in [1,5,20,75,250]:
 p=np.column_stack([np.mean([preds[i][:,j] for i in np.argsort(S[:,j])[:k]],axis=0) for j in range(4)]);print('ensemble',k,abs(p-Y[n:]).mean(0),abs(p[:,1:]@H.T-tr[n:]).mean())
