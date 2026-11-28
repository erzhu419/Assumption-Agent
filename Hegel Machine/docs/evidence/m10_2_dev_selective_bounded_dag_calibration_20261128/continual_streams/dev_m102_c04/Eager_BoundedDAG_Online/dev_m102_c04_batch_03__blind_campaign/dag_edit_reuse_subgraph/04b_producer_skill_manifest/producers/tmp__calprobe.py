exec(open('/tmp/modelprobe.py').read().split('o,M=')[0]); pop,rop=m['_panel_operators'](C)
u,v=A['u'],A['v']; mask=np.ones(len(u),bool)
def robust(design):
 h=design(u); mask=np.ones(len(u),bool)
 for _ in range(15):
  w=ls(h[mask],v[mask]); e=np.linalg.norm(h@w-v,axis=1); mask=e<max(4*np.median(e),.002)
 return w,e,mask
w,e,mask=robust(a); print('full A',w,'error',np.sort(e)); bo,*bm=ls(a(B['u']),B['v']); bm=np.array(bm); o,M=w[0],w[1:]; print('A sv',np.linalg.svd(M)[1]); print('A inliers ranges',u[mask].min(0),u[mask].max(0),'study ranges',x.min(0),x.max(0))
q=np.linalg.solve(M.T,(x-o).T).T
coords={'x':x,'q':q,'qB':q@bm+bo,'qBinv':np.linalg.solve(bm.T,(q-bo).T).T,'xB':x@bm+bo,'xBinv':np.linalg.solve(bm.T,(x-bo).T).T}; coords['invAxBinv']=np.linalg.solve(M.T,(coords['xBinv']-o).T).T
scores=temporal['scores']; target=np.column_stack((y[:n],scores)); ans=[]
for pn,pq in coords.items():
 for rn,rq in coords.items():
  p,r=pop(pq),rop(rq); diag=p*r; cross=np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4) if i!=j]);
  for label,c in [('dot',np.c_[p,r,diag.sum(1)]),('diag',np.c_[p,r,diag]),('all',np.c_[p,r,diag,cross]),('linear',np.c_[p,r])]:
   yy=ridge(c[:n],y[:n],c[n:],1e-6); out=frozen({'node_00':dict(common=c,offdiag=cross,y_hat=yy),'node_01':temporal}); mae=np.mean(abs(yy-y[n:])); tm=np.mean(abs(out['state_pred']-traj[n:])); trm=np.mean(abs(a(c[:n])@ls(a(c[:n]),target)-target),axis=0); ans.append((mae+tm,mae,tm,pn,rn,label,trm))
print('best',*sorted(ans,key=lambda z:z[0])[:20],sep='\n'); print('archived feature in full A',[(m,t,p,r,l,trm) for _,m,t,p,r,l,trm in ans if p==r=='q'])
for name,design in [('square',lambda z: np.c_[a(z),z*z]),('pairwise',lambda z: np.c_[a(z),z[:,np.triu_indices(4)[0]]*z[:,np.triu_indices(4)[1]]]),('panel',lambda z: np.c_[a(z),pop(z),rop(z)])]:
 w,e,mask=robust(design); print(name,np.quantile(e,[0,.25,.5,.75,.90,.93,.95,1]))
