import numpy as np, pickle, importlib.util,itertools
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; spec=importlib.util.spec_from_file_location('enc',P+'_frozen_nodes/node_00.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
z=m.run_node({'__input__':s})['z_tr']; x=s['train']['x']; a=s['panel_a']; u=m._invert_a(x,*m._fit_a(a['u'],a['v'])); Y=np.c_[s['train']['y'],s['train']['traj']]; fs=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),i), i) for i in np.array_split(np.random.default_rng(123).permutation(72),4)];
def poly(q,d):return np.column_stack([np.prod(q[:,j],axis=1) for n in range(d+1) for j in itertools.combinations_with_replacement(range(q.shape[1]),n)])
def check(D,Y=Y):
 es=np.array([np.mean(abs(D[b]@np.linalg.lstsq(D[a],Y[a],rcond=None)[0]-Y[b]),axis=0) for a,b in fs]); return np.round(np.c_[es[:,0],es[:,1:].mean(axis=1)][[0,1+np.argmax(es[1:,0]),1+np.argmax(es[1:,1:].mean(axis=1))]],5).tolist()
np.set_printoptions(precision=4,suppress=True);print('z range',z.min(0),z.max(0));print('z rows',np.c_[z,Y[:,0],Y[:,-1]][:12]);
for name,q in [('encoded',z),('reference',u),('measured',x)]:
 for d in [1,2,3,4]: print(name,d,check(poly(q,d)))
 for fun in [np.tanh,np.sin,lambda q:q*np.abs(q),lambda q:q/(1+q*q),lambda q:q*np.exp(-q*q)]:
  for k in [.5,1,2]: print(name,fun.__name__,k,check(np.c_[poly(q,2),fun(k*q)]))
D=poly(z,3); c=np.linalg.lstsq(D,Y,rcond=None)[0]; print('encoded cubic terms: monomial, y coefficient, final trajectory coefficient');print(list(zip([j for n in range(4) for j in itertools.combinations_with_replacement(range(4),n)],c[:,0].round(4),c[:,-1].round(4))))
