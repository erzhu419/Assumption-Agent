import pickle,runpy,numpy as np
m=runpy.run_path('/holdout_task/node_00.py'); a,ia,b,bb=[m[x] for x in ('fit_a','inv_a','_fit_b','_basis_b')]; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; C=s['panel_c']; ac=a(s['panel_a']); bp,bc=b(s['panel_b']); B=lambda x:bb(x,bp)@bc; A=lambda x:ac[0]+ac[1]*x+ac[2]*(x-1)*np.roll(x,-1,axis=1)+ac[3]*x**3
np.set_printoptions(precision=4,suppress=True); print('B parameters',bp,bc); print('C keys',[(k,v.shape,v.min(0),v.max(0)) for k,v in C.items()]); print('C means',[(k,v.mean(0)) for k,v in C.items()]); print('C rows',[(k,v[:3]) for k,v in C.items()]); u=C['u']; t=ia(s['train']['x'],ac); Y=s['train']['traj']; aff=lambda x:np.c_[np.ones(len(x)),x]; quad=lambda x:np.c_[aff(x),*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]]
for key,v in C.items():
 if key=='u' or v.ndim!=2 or v.shape[1]!=4:continue
 for tag,f in [('A',A),('B',B),('AB',lambda x:A(B(x))),('BA',lambda x:B(A(x))),('identity',lambda x:x)]:print(key,tag,np.sqrt(np.mean((f(u)-v)**2)))
 for mode,feat in [('aff',aff),('quad',quad),('B-aff',lambda x:aff(B(x))),('B-quad',lambda x:quad(B(x)))]:
  X=feat(u); w=np.linalg.lstsq(X,v,rcond=None)[0]; e=X@w-v; h=np.sum(np.linalg.qr(X)[0]**2,axis=1); Z=aff(feat(t)@w); k=len(Y)-20; er=Z[k:]@np.linalg.lstsq(Z[:k],Y[:k],rcond=None)[0]-Y[k:]; print(key,mode,np.sqrt(np.mean(e**2)),np.mean(abs(e/(1-h)[:,None])),np.mean(abs(er)))
