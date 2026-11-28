import pickle,sys,itertools,numpy as np
sys.path.insert(0,'/holdout_task'); import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; q=n._inverse_a(tr['x'],n._fit_panel_a(s['panel_a'])); b=n._fit_panel_b(s['panel_b']); M=b[1:]; t=tr['traj']; y=tr['y']; pairs=list(itertools.combinations_with_replacement(range(4),2)); aff=lambda x:np.column_stack((np.ones(len(x)),x)); quad=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]);
for name,z in [('q',q),('Bf',q@M+b[0]),('Bi',(q-b[0])@np.linalg.inv(M))]:
 X=np.column_stack((aff(z),quad(z))); c=np.linalg.lstsq(X,t,rcond=None)[0]; U,S,V=np.linalg.svd(c[5:],full_matrices=False); print(name,'range',np.round([z.min(0),z.max(0)],3),'qsvals',np.round(S,5));print('quad dirs',list(zip(pairs,np.round(U[:,:3],3).tolist())));print('lin',np.round(c[1:5,[-1]],3).ravel());
for k in ('panel_a','panel_b','panel_c'):
 p=s[k];print(k,'bounds',[(j,np.round(np.min(v,0),2).tolist(),np.round(np.max(v,0),2).tolist()) for j,v in p.items()]);
# Shared carrier tests: fit on first 52 and independently report remaining trajectory errors.
C=s['panel_c']; u=C['u']; v=C['v2']; best=None
for f in np.linspace(.2,4,191):
 F=lambda a:np.column_stack((aff(a),a*a,np.cos(f*a))); e=np.linalg.lstsq(F(u),v,rcond=None)[0]; loss=np.linalg.norm(F(u)@e-v);
 if best is None or loss<best[0]:best=(loss,f,e)
_,f,e=best; C2=lambda a:np.column_stack((aff(a),a*a,np.cos(f*a)))@e
qf=q@M+b[0]; qi=(q-b[0])@np.linalg.inv(M); cross=lambda a:quad(a)[:,[k for k,(i,j) in enumerate(pairs) if i!=j]]
carriers={'q cross':cross(q),'Bf cross':cross(qf),'Bi cross':cross(qi),'q diag':q*q,'Bf diag':qf*qf,'Bi diag':qi*qi,'C2(q)':C2(q),'C2(Bf)':C2(qf),'C2(Bi)':C2(qi),'q01+q23':q[:,0]*q[:,1]+q[:,2]*q[:,3]}
for i,(a,A) in enumerate(carriers.items()):
 for j,(d,D) in enumerate(carriers.items()):
  if j<i:continue
  X=np.column_stack((aff(q),A,D)); E=np.mean(abs(X[52:]@np.linalg.lstsq(X[:52],t[:52],rcond=None)[0]-t[52:])); print('carrier',round(E,5),a,d)
