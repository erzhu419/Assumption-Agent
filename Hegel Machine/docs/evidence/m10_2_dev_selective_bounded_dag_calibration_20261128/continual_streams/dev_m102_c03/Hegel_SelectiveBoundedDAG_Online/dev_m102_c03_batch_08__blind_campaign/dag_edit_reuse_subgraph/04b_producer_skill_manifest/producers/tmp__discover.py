import numpy as np,pickle,sys,itertools,importlib.util
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; t=s['train']['traj']; n=len(y); times=s['eval_times']; h=f.run_node({'__input__':s})['hy'][:n]; c,d=h[:,1:5],h[:,5:9]; z=f._inverse(s['train']['x'],f._calibration(s['panel_a'])); p=s['panel_b']; b=f._one(z)@f._fit(f._one(p['u']),p['v']); e=f._edge(z); H=np.c_[h,c[:,0]*d[:,1]]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
Y=np.c_[y,t]; R=Y-H@fit(H,Y); print('linear residual slope',fit(R[:,0,None],R[:,1:]).round(4)); print('joint residual',np.abs(R[:,1:]-R[:,0,None]*np.sqrt(times)*.5).mean(0)); print('panel b lin error',np.std(f._one(p['u'])@f._fit(f._one(p['u']),p['v'])-p['v']));
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/dev_m102_c03_batch_06_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
for tag,A in [('base',H),('old',m._design(s,s['train']['x'])),('z',np.c_[H,z]),('z cubic',np.c_[H,*[np.prod(z[:,k],axis=1) for j in range(1,4) for k in itertools.combinations_with_replacement(range(4),j)]]),('bcross',np.c_[H,(b[:,:,None]*c[:,None,:]).reshape(n,-1)])]:
 p=A[52:]@fit(A[:52],Y[:52]); print(tag,np.abs(p-Y[52:]).mean(0).round(4),np.abs(Y-A@fit(A,Y)).mean(0).round(4))
D={}
for k,a in [('c',c),('d',d),('z',z),('b',b),('e',e)]:
 for i,x in enumerate(a.T): D[k+str(i)]=x
 D[k+'norm']=np.linalg.norm(a,axis=1); D[k+'sum']=a.sum(1)
D['dot']=h[:,-1]; D['cd01']=c[:,0]*d[:,1]; seeds=dict(D)
for (a,x),(b,v) in itertools.combinations(list(seeds.items()),2):
 D[a+'*'+b]=x*v; D[a+'+'+b]=x+v; D[a+'-'+b]=x-v; D[a+'/(1+'+b+'^2)']=x/(1+v*v); D[b+'/(1+'+a+'^2)']=v/(1+x*x)
for a,x in seeds.items():
 for b,v in seeds.items():
  if b>=a: continue
  for k,w in seeds.items():
   if k>b: continue
   D[a+'*'+b+'*'+k]=x*v*w
for a,x in list(D.items()):
 if a.count('*')<2:
  for j in (.5,1,2): D['tanh'+str(j)+'('+a+')']=np.tanh(j*x);D['sin'+str(j)+'('+a+')']=np.sin(j*x);D['cos'+str(j)+'('+a+')']=np.cos(j*x)
keys=list(D); A=np.column_stack(list(D.values())); A-=H@fit(H,A); norms=np.linalg.norm(A,axis=0); cc=np.abs(R[:,0]@A)/np.maximum(norms,1e-12)/np.linalg.norm(R[:,0]); print('top',[(keys[j],round(cc[j],4)) for j in np.argsort(cc)[-30:][::-1]]); k=np.argmax(cc); print('best CV',np.abs(Y[52:]-np.c_[H,A][52:][:,list(range(H.shape[1]))+[H.shape[1]+k]]@fit(np.c_[H,A][:52][:,list(range(H.shape[1]))+[H.shape[1]+k]],Y[:52])).mean(0))
