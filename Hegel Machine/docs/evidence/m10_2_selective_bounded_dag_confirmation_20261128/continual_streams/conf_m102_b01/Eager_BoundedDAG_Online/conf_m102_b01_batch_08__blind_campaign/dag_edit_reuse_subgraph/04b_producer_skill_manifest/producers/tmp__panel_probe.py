import numpy as np,pickle,itertools,runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('telemetry',s['telemetry_vec']); a=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s});print('frozen',[(k,np.shape(v)) for k,v in a.items()]);
for p,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][key];print(p,key)
 for kind in ['linear','quadratic','channel','cyclic','cubic']:
  r=[];err=[]
  for j in range(4):
   q=(j+1)%4;t=(j+2)%4
   if kind in ['linear','quadratic','cubic']:
    deg={'linear':1,'quadratic':2,'cubic':3}[kind];d=np.column_stack([np.ones(len(u))]+[np.prod(u[:,ix],axis=1) for m in range(1,deg+1) for ix in itertools.combinations_with_replacement(range(4),m)])
   elif kind=='channel':d=np.column_stack([np.ones(len(u)),u[:,j],u[:,j]**2,u[:,j]**3])
   else:d=np.column_stack([np.ones(len(u)),u,u[:,j]*u[:,q],u[:,j]*u[:,t],u[:,j]**2,u[:,j]**3])
   c=np.linalg.lstsq(d,v[:,j],rcond=None)[0]; e=d@c-v[:,j];r.append(np.sqrt(np.mean(e**2)));err.append(e)
  print(kind,len(c),np.round(r,5),np.round(np.sort(np.linalg.norm(err,axis=0))[-6:],5))
