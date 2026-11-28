exec(open('/tmp/inspect03.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.max(abs(v-u),axis=1)<3
pol=PolynomialFeatures(3); FF=pol.fit_transform(u); M=np.linalg.lstsq(FF[keep],v[keep],rcond=None)[0]; print('A terms',[(k,np.round(m,4).tolist()) for k,m in zip(pol.get_feature_names_out(),M) if max(abs(m))>.001]); x=tr['x']; z=least_squares(lambda z:(pol.transform(z.reshape(-1,4))@M-x).ravel(),(x/1.2).ravel(),gtol=1e-10).x.reshape(-1,4)
p=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; bz=lambda z:z@B[1:]+B[0]; A=b.run_node({'__input__':s})['basis']; Y=np.c_[tr['y'],tr['traj']@np.linalg.pinv(A).T]; print('inversion',np.max(abs(pol.transform(z)@M-x))); Q=[]
for name,w in [('z',z),('bz',bz(z)),('iz',(z-B[0])@np.linalg.inv(B[1:])),('cz',a._panel_maps(s['panel_c'],z)),('cbz',a._panel_maps(s['panel_c'],bz(z))),('ciz',a._panel_maps(s['panel_c'],(z-B[0])@np.linalg.inv(B[1:])) )]:
 for deg in (1,2,3):
  for block in ([w] if w.shape[1]==4 else [w,w[:,:4]]):
   F=PolynomialFeatures(deg).fit_transform(block); pp=[]
   for seed in [-1,0,1,2]:
    idx=np.arange(n) if seed<0 else np.random.default_rng(seed).permutation(n); i,j=idx[:54],idx[54:]; U,D,V=np.linalg.svd(F[i],full_matrices=False); K=F[j]@V.T
    pp.append([[np.mean(abs(d[:,0])),np.mean(abs(d[:,1:]@A.T))] for r in (1e-8,.01,.1) for d in [(K*(D/(D*D+r)))@(U.T@Y[i])-Y[j]]])
   Q.append((name,deg,block.shape[1],np.round(np.mean(pp,axis=0),4).tolist(),np.round(np.max(pp,axis=0),4).tolist()))
print('results'); print(*Q,sep='\n')
