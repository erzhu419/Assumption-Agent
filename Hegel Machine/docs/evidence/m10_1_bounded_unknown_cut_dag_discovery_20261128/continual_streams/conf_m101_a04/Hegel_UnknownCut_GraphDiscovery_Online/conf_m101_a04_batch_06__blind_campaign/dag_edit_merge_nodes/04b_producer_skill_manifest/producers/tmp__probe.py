import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_01
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=node_00.run_node({'__input__':s}); x=p['x']; b=node_01.run_node({'node_00':p})['b']; n=len(s['train']['y']); tr=s['train']['traj']; y=s['train']['y']; u=p['u'];
np.set_printoptions(precision=6,suppress=True)
print('ranges',[(a.min(),a.max()) for a in (x,b,p['v2'],tr,y)]);print('traj sample',tr[:7]);print('traj svd',np.linalg.svd(tr,compute_uv=False));print('diff svd',np.linalg.svd(np.diff(tr,axis=1),compute_uv=False));print('y traj pairs',np.c_[y,tr[:,0],tr[:,-1]][:12]);print('panel b keys',s['panel_b'].keys())
def lib(u):
 d={'1':np.ones(len(u))};m=u.shape[1]
 for k in range(1,4):
  for ix in itertools.combinations_with_replacement(range(m),k):d[''.join(str(i) for i in ix)]=np.prod(u[:,ix],axis=1)
 for i in range(m):
  a=u[:,i];d[f'tanh{i}']=np.tanh(a);d[f'sin{i}']=np.sin(a);d[f'cos{i}']=np.cos(a);d[f'sat{i}']=a/(1+a*a);d[f'abs{i}']=abs(a)
  for j in range(i+1,m):
   z=u[:,j];d[f'sin{i}+{j}']=np.sin(a+z);d[f'sin{i}-{j}']=np.sin(a-z);d[f'tanh{i}+{j}']=np.tanh(a+z);d[f'tanh{i}-{j}']=np.tanh(a-z);d[f'tanh{i}*{j}']=np.tanh(a*z);d[f'tanh{i}x{j}']=np.tanh(a)*z;d[f'{i}xtanh{j}']=a*np.tanh(z);d[f'sat{i},{j}']=a/(1+z*z);d[f'sat{j},{i}']=z/(1+a*a)
 return d
def omp(d,v,k=12):
 names=list(d);a=np.column_stack(list(d.values())); norm=np.linalg.norm(a,axis=0);r=v.copy();sel=[]
 for _ in range(k):
  z=np.abs(a.T@r)/norm;z[sel]=0;sel.append(np.argmax(z));c=np.linalg.lstsq(a[:,sel],v,rcond=None)[0];r=v-a[:,sel]@c
  if np.linalg.norm(r)<1e-9:break
 return ' '.join(names[i] for i in sel),np.round(c,5),np.sqrt(np.mean(r*r))
for name,a,v in [('c2',u,p['v2']),('b',s['panel_b']['u'],s['panel_b']['v'])]:
 for j in range(v.shape[1]):print(name,j,omp(lib(a),v[:,j]))
for name,a in [('x',np.c_[np.ones(n),x[:n]]),('x+b',np.c_[np.ones(n),x[:n],b[:n]]),('x-poly',np.column_stack(list(lib(x[:n]).values())[:35])),('traj',np.c_[np.ones(n),tr])]:
 for name2,v in [('y',y),('tr',tr)]:
  c=np.linalg.lstsq(a[:52],v[:52],rcond=None)[0];print(name,name2,np.mean(abs(a[52:]@c-v[52:])),np.mean(abs(a[:52]@c-v[:52])))
