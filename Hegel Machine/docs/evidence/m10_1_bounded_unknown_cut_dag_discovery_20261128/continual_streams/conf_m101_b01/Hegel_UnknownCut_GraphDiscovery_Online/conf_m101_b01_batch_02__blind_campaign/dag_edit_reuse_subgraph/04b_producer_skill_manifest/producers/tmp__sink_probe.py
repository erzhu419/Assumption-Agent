import pickle,numpy as np,itertools
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; t=s['train']; u=p['u']; v=p['v2']
def design(*xs): return np.column_stack((np.ones(len(xs[0])),*xs))
def fit(f,v): return np.linalg.lstsq(f,v,rcond=None)[0]
def check(name,f,v):
 w=fit(f,v); e=v-f@w; loo=e/(1-(f*np.linalg.pinv(f).T).sum(1)).reshape((-1,)+(1,)*(e.ndim-1)); print(name,f.shape[1], 'rmse',np.sqrt(np.mean(e*e,axis=0)), 'loo mae',np.mean(abs(loo),axis=0));return w
pairs=((0,1),(0,3),(1,2),(2,3))
def cross(z): return np.column_stack([z[:,i]*z[:,j] for i,j in pairs])
print('panel ranges',u.min(0),u.max(0))
for i in range(1,6):
 w=check('panel additive powers '+str(i),design(*[u**k for k in range(1,i+1)]),v)
 if 2<=i<=4: print(w)
for a in (.25,.35,.45,.5,.55,.65,1):
 for name,f in [('expm1',np.expm1(a*u)),('sin+quad',np.sin(u)+a*u*u),('tanh+quad',np.tanh(u)+a*u*u),('quad',u+a*u*u)]: check((name,a),design(f),v)
for i in (0,1):
 ns={};exec(open('/holdout_task/_frozen_nodes/node_0%d.py'%i).read(),ns);out=ns['run_node']({'__input__':s} if i==0 else {'node_00':out})
z=out['x'][:len(t['y'])]; c=design(cross(z))@fit(design(cross(u)),p['v1']); w=fit(design(u,u*u,u**3,u**4),v); b=design(z,z*z,z**3,z**4)@w
print('stage range',z.min(0),z.max(0),'stage v1 range',c.min(0),c.max(0),'stage v2 range',b.min(0),b.max(0))
y=t['y']; tr=t['traj']; print('traj singvals',np.linalg.svd(tr,compute_uv=False)); print('traj means',tr.mean(0));
for name,f in [('z',z),('v1',c),('v2',b),('v1v2',np.c_[c,b]),('z+v1+v2',np.c_[z,c,b]),('tanh(v1)',np.tanh(c)),('tanh(v2)',np.tanh(b)),('z powers 4',np.c_[z,z*z,z**3,z**4]),('v1 powers 4',np.c_[c,c*c,c**3,c**4]),('v2 powers 4',np.c_[b,b*b,b**3,b**4])]:
 print('train',name); wy=check('y',design(f),y); wt=check('traj',design(f),tr); print('wy',wy) if len(wy)<=9 else None
