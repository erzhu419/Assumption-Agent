import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/compare_maps.py').read())
np.set_printoptions(precision=4,suppress=True,linewidth=135)
def loo(F,V):
 c=np.linalg.lstsq(F,V,rcond=None)[0]; e=V-F@c; h=np.sum(F*np.linalg.pinv(F).T,axis=1); return np.linalg.norm(e/(1-h[:,None]),axis=1),c,e
u=(x-make(a['u'],a['v'],1,True)[1][0])@np.linalg.inv(make(a['u'],a['v'],1,True)[1][1:]);
print('latent',np.quantile(u,[0,.5,1],axis=0),'y quantiles',np.quantile(train['y'],[0,.2,.5,.8,1]),'traj svals',np.linalg.svd(train['traj'],compute_uv=False))
for name,U,V in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2']),('C1->2',c['v1'],c['v2']),('response',u,np.c_[train['y'],train['traj']@np.linalg.svd(train['traj'],full_matrices=False)[2][:2].T])]:
 F=features(U,1); q=features(U,2)[:,5:]; L=loo(F,V); Q=loo(np.c_[F,q],V)
 print('\n',name,'affine matrix',L[1],'quad',Q[1][5:],'res sv',np.linalg.svd(L[2],compute_uv=False)); print('sorted input norms with residual',np.c_[np.linalg.norm(U,axis=1),np.linalg.norm(L[2],axis=1)].round(4) if name=='B' else '');
 cand=[]
 def trial(n,T):
  e,cf,r=loo(np.c_[F,T],V); cand.append((np.mean(e),n,np.max(e),np.linalg.norm(r)/len(V)**.5))
 trial('affine',np.zeros((len(U),0)));trial('quad',q);trial('squares',U**2);trial('cross',q[:,[1,2,3,5,6,8]]); trial('cubes',U**3)
 for k in [.25,.5,1,1.25,1.5,2,2.5,3,3.5,4,5,6,8]:
  for n,T in [('sin',np.sin(k*U)),('cos',np.cos(k*U)),('tanh',np.tanh(k*U)),('sin+cos',np.c_[np.sin(k*U),np.cos(k*U)]),('sinsum',np.column_stack([np.sin(k*(U[:,i]+sgn*U[:,j])) for i in range(4) for j in range(i+1,4) for sgn in [-1,1]])),('sinprod',np.sin(k*q[:,[1,2,3,5,6,8]])),('radial',U/(1+k*(U**2).sum(1)[:,None])),('compress',U/(1+k*U**2)),('cosrot',np.cos(k*(F@L[1]))),('sinrot',np.sin(k*(F@L[1])))]:trial(f'{n} {k}',T)
 print('best LOO mean max rmse:', sorted(cand)[:12]); print('base:',[t for t in cand if not any(ch.isdigit() for ch in t[1])]);
