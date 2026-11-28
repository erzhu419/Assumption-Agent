exec(open('/tmp/fit07.py').read().split('for name,b')[0])
from scipy.linalg import svd
W=w[:n]; ring=lambda x:x*np.roll(x,-1,axis=1)
R=ring(W); Q=reg(np.c_[np.ones(len(u)),ring(u)],p['v1'],0); V=np.c_[np.ones(n),R]@Q
print('ring panel RMSE',np.std(np.c_[np.ones(len(u)),ring(u)]@Q-p['v1']))
U,D,T=svd(tr,full_matrices=False); pc=U*D; print('time vectors',T[:3]); print('small/big',np.c_[y,pc[:,:3]][np.argsort(abs(pc[:,0]))[::4]]); print('corr ring y/traj',np.corrcoef(np.c_[R,V,y,pc[:,:3]].T)[-4:,:8]); print('range R,V',R.min(0),R.max(0),V.min(0),V.max(0))
AA={'W':W,'rawx':s['train']['x'],'ring':R,'v1':V,'center_v1':V-Q[0],'ring_rawx':ring(s['train']['x']),'ring_panelb_inv':ring(runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['_a'](s['panel_a']['u'],s['panel_a']['v'],s['train']['x']))}
fs=[]
for key,Z in AA.items():
 for k in list(fun)+['sinh','cube','signedsqrt','logabs','cos','arctan2','abs','expneg']:
  f={**fun,'sinh':np.sinh,'cube':lambda z:z**3,'signedsqrt':lambda z:np.sign(z)*np.sqrt(abs(z)),'logabs':lambda z:np.log1p(abs(z)),'cos':np.cos,'arctan2':lambda z:np.arctan(z*z),'abs':np.abs,'expneg':lambda z:np.exp(-z*z)}[k]
  for g in [.45,1,2,4,8,16]:
   F=f(g*Z)
   for mode,A in enumerate([F,np.c_[Z,F],np.c_[Z,F,F*F],np.c_[R,F],np.c_[R,F,pair(F)],np.c_[R,F,W],np.c_[poly(R,2),F],np.c_[poly(R,2),F,F**2],np.c_[poly(R,2),F,W],np.c_[Z,F,poly(Z,2)]]):
    if abs(A).max()>1e14:continue
    fs.append(((key,k,g,mode),np.c_[np.ones(n),A]))
for name,b in [('y',y),('tr',tr)]:
 out=[]
 for label,A in fs:
  for r in [0,.25,2]:
   ers=[np.mean(abs(A[v]@reg(A[a],b[a],r)-b[v])) for a,v in folds]; out.append((np.mean(ers),max(ers),ers[0],label,r,np.mean(abs(A@reg(A,b,r)-b))))
 print(name,*sorted(out)[:18],sep='\n')
