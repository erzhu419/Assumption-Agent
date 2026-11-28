import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import subspace_angles
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,T=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=115)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def lin(x):return np.column_stack([np.ones(len(x)),x])
def quad(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)])
u,v=(s['panel_a'][k] for k in ('u','v')); A=[]
for i in range(4):
 h=lin(u[:,i]); init=np.polyfit(u[:,i],v[:,i],1); good=np.argsort(abs(h@init[::-1]-v[:,i]))[:42]; w=ls(h[good],v[good,i]); err=abs(h@w-v[:,i]); good=err<max(.001,5*np.median(err)); A.append(ls(h[good],v[good,i]))
a,d=np.asarray(A).T; B=ls(lin(s['panel_b']['u']),s['panel_b']['v']); C=s['panel_c']; C2=ls(np.c_[lin(C['u']),C['u']**2],C['v2']); z=x*d+a; bases={'x':x,'A_forward':z,'A_inverse':(x-a)/d,'AB':lin(z)@B,'XB':lin(x)@B,'AiB':lin((x-a)/d)@B,'ABt':(z-B[0])@B[1:].T,'ABinv':(z-B[0])@np.linalg.inv(B[1:])}; print('A',a,d,'B',B,'C2',C2)
from scipy.optimize import least_squares
z=bases['AiB']; L=lin(z); p=z[:,0]*z[:,2]; q=z[:,1]*z[:,3]; XX={'full':np.c_[L,quad(z)],'pair':np.c_[L,p,q],'shared':np.c_[L,p+q],'fixed18':L};
def expo(params,t): return -np.expm1(-np.exp(params)[None,:]*t[:,None])
def fit_exp(L,T):
 C=ls(L,T); P=np.linalg.qr(L,mode='reduced')[1]@C
 def res(v):
  E=expo(v,t); return (ls(E,P.T).T@E.T-P).ravel()
 r=least_squares(res,np.log([.4,1.8]),bounds=(np.log(.03),np.log(15)),max_nfev=100); E=expo(r.x,t)
 return ls(E,C.T).T,E,np.exp(r.x)
print('scalar paired',ls(XX['pair'],y),'shared', ls(XX['shared'],y)); print('trajectory modes',fit_exp(L,T));
rng=np.random.default_rng(988); splits=[('56/16',np.arange(56,72))]; splits.extend((f'block{i}',np.arange(i,i+12)) for i in range(0,72,12)); splits.extend((f'max{i}',np.argsort(z[:,i])[-12:]) for i in range(4)); splits.extend((f'min{i}',np.argsort(z[:,i])[:12]) for i in range(4)); splits.extend((f'random{i}',ix) for i,ix in enumerate(np.array_split(rng.permutation(len(z)),6)))
metrics={k:[] for k in [*XX,'affine','rank2','exp2']};
for name,j in splits:
 i=np.setdiff1d(np.arange(len(z)),j)
 for k,X in XX.items():
  offset=.18*(p+q) if k=='fixed18' else np.zeros(len(z)); pred=X[j]@ls(X[i],(y-offset)[i])+offset[j]; metrics[k].append(np.mean(abs(y[j]-pred)))
 C=ls(L[i],T[i]); _,_,V=np.linalg.svd(L[i]@C,full_matrices=False); V=V[:2]; W,E,_=fit_exp(L[i],T[i]);
 for k,pred in zip(('affine','rank2','exp2'),(L[j]@C,L[j]@C@V.T@V,L[j]@W@E.T)):
  metrics[k].append(np.mean(abs(pred-T[j]),axis=0));
for k,m in metrics.items():
 m=np.array(m); groups=[slice(0,1),slice(1,7),slice(7,15),slice(15,None)]; print(k,'first / blocks / extremal / random avg,worst',[(round(m[g].mean(),6),round(m[g].max(),6)) for g in groups],'worst slice',splits[np.unravel_index(m.argmax(),m.shape)[0]][0]);
