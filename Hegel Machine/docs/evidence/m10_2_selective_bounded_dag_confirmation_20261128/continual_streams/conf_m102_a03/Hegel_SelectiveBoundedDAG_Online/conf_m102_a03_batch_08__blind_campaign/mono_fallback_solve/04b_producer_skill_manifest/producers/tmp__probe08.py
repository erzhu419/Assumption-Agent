import sys,glob,pickle,numpy as np,importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; S=dict(s,train=dict(s['train'],x=x[:52],y=y[:52],traj=t[:52]),test={'x':x[52:]})
for p in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('old',p));sp.loader.exec_module(m)
 try:
  a=m.solve_batch(S); print(p, np.mean(abs(a['y_hat']-y[52:])), np.mean(abs(a['state_pred']-t[52:])))
 except Exception as e: print(p,e)
 if '06_solver' in p:
  z=m.calibrate(s['panel_a'],x); b=m.branch(s['panel_b'],z); B,f=m.sine_branch(s['panel_b'],z); print('branch',f,'range',z.min(),z.max()); np.savez('/tmp/feat08.npz',x=x,z=z,b=b,B=B,y=y,t=t)
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v,axis=1)<5; print('keep',sum(keep),np.sort(np.linalg.norm(v,axis=1))[-9:]);
for j in range(4):
 a=u[:,j]; b=u[:,(j+1)%4]; M=np.c_[np.ones(len(u)),u,a*b,a**3]; c=np.linalg.lstsq(M[keep],v[keep,j],rcond=None)[0]; print('cal',c,np.std((M@c-v[:,j])[keep]))
print('traj sv',np.linalg.svd(t,compute_uv=False))
