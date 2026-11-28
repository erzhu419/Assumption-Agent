exec(open('/tmp/probe.py').read().split("for f in sorted")[0])
import importlib.util
p='/tmp/hegel_continual_archive/conf_m102_a03_batch_02_solver.py'; spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=p.replace('02_solver','03_solver'); spec=importlib.util.spec_from_file_location('other',p); o=importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
c=m._calibration(s['panel_a']); z=m._decode(tr['x'],c); params,mix=o._operator(s['panel_b']); w=affine(o._warp(z,params))@mix
print('calibration row errors',np.sort(np.linalg.norm((c[:4]+c[4:8]*s['panel_a']['u']+c[8]*np.roll(s['panel_a']['u'],-1,axis=1)+c[9]*s['panel_a']['u']*np.roll(s['panel_a']['u'],-1,axis=1)+c[10]*s['panel_a']['u']**3)-s['panel_a']['v'],axis=1)))
y,T=tr['y'],tr['traj']; t=s['eval_times']; cross=np.array([w[:,i]*w[:,j] for i in range(4) for j in range(i,4)]).T; h=w[:,0]*w[:,2]+w[:,1]*w[:,3]
for name,a in [('lin',w),('aff',affine(w)),('prior bilinear',np.c_[w,h]),('prior bilinear affine',np.c_[affine(w),h]),('all quad',np.c_[affine(w),cross])]:
 print(name,'y',score(a,y),'T',score(a,T),'coef y',fit(a[:52],y[:52])); print('T first/last coeffs',fit(a[:52],T[:52])[:,[0,-1]].T)
for name,A in [('lin',w),('aff',affine(w))]:
 Q=np.linalg.qr(A[:52],mode='reduced')[0]; projected=(Q.T@T[:52]).T
 for typ in ['sat','exp','sat+exp','sat+time','sat+timesat']:
  def basis(p):
   e=np.exp(-np.outer(t,p)); sat=1-e
   return {'sat':sat,'exp':e,'sat+exp':np.c_[sat[:,0],e[:,1]],'sat+time':np.c_[sat[:,0],t],'sat+timesat':np.c_[sat[:,0],t*sat[:,1]]}[typ]
  def res(p):
   B=basis(p);return (B@fit(B,projected)-projected).ravel()
  r=least_squares(res,[.45,1.7],bounds=([.001,.001],[12,12]),gtol=1e-12,ftol=1e-12,xtol=1e-12);B=basis(r.x); full=A@fit(A[:52],T[:52])@B@np.linalg.pinv(B)
  print(name,typ,r.x,np.linalg.norm(r.fun),np.mean(abs(full[52:]-T[52:])),np.mean(abs(full[:52]-T[:52])))
