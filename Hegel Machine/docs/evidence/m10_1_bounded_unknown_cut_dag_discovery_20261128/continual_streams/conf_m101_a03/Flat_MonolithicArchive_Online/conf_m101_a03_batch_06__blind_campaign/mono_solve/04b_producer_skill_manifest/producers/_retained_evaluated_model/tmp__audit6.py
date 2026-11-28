import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._calibration(s['panel_a'],tr['x']); y,T=tr['y'],tr['traj']; print('study keys',list(s),'train keys',list(tr),'telemetry shape',np.shape(s['telemetry_vec']))
a=s['panel_a']; u,v=a['u'],a['v']; keep=np.argsort(np.linalg.norm(v-u,axis=1))[:-3]; eye=np.broadcast_to(np.eye(4),(len(u),4,4)); design=np.concatenate((eye,eye*u[:,:,None],((1+u)*np.roll(u,-1,axis=1))[:,:,None],(u**3)[:,:,None]),axis=2)
for d in [8,10]:
 err=[]
 for j in keep:
  i=keep[keep!=j]; p=m._fit(design[i,:,:d].reshape(-1,d),v[i].ravel()); err.extend(abs(design[j,:,:d]@p-v[j]))
 print('calibration terms',d,'LOO mean/max',np.mean(err),max(err))
coords={'A':z}; b=s['panel_b']; coords['B']=m._affine(z)@m._fit(m._affine(b['u']),b['v']); c=s['panel_c']; q=m._quadratic(c['u']);
for key in ('v1','v2'):
 w=m._fit(q,c[key]); print('panel C',key,'residual RMS',np.std(q@w-c[key]),'coefficients',w.round(3).T); coords['C_'+key]=m._quadratic(z)@w;coords['BC_'+key]=m._quadratic(coords['B'])@w
rows=np.arange(len(y)); norm=np.argsort(np.linalg.norm(z,axis=1)); splits=[('52/20',rows[:52],rows[52:]),('outer',norm[:-14],norm[-14:])]
for name,i,j in splits:
 for k,h in coords.items():
  q=m._quadratic(h); errors=[]
  for link in ('tanh','rat') if k=='A' else ('tanh',):
   p,w,f=m._bounded_fit(h[i],y[i],link,starts=12); errors.append(np.mean(abs(np.c_[q[j],f(m._affine(h[j])@p)]@w-y[j])))
  print(name,k,'scalar MAE',np.round(errors,6),'trajectory MAE',round(np.mean(abs(q[j]@m._fit(q[i],T[i])-T[j])),6))
