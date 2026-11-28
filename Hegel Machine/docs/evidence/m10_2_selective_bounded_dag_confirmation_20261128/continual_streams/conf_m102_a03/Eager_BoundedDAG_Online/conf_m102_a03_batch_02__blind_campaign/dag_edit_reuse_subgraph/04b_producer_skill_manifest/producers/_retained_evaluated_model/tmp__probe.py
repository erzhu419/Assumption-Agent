import runpy,pickle,numpy as np,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z']; y=s['train']['y']; a=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
blocks={'Z':z,'X':s['train']['x']}
for k,v in [('B','v'),('C','v1'),('C','v2')]:
 p=s['panel_'+k.lower()]; b=fit(a(p['u']),p[v]); print(k,v,'affine residual',np.max(abs(a(p['u'])@b-p[v])), '\n',b)
 if k=='B': u=np.linalg.solve(b[1:].T,(z-b[0]).T).T;blocks['U']=u
 else: blocks[v]=a(u)@b
order=np.random.default_rng(42).permutation(len(y)); splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(order,J),J) for J in np.array_split(order,6)]
def met(D):
 e=[np.abs(D[j]@fit(D[i],y[i])-y[j]).mean() for i,j in splits];return np.round([e[0],np.mean(e[1:]),max(e[1:]),np.mean(abs(D@fit(D,y)-y))],5)
for name,x in blocks.items():
 print(name,'ranges',x.min(0),x.max(0)); D=a(x)
 for k in [1,2,3,4]:
  if k>1: D=np.c_[D,*[np.prod(x[:,p],axis=1) for p in it.combinations_with_replacement(range(4),k)]]
  print('degree',k,met(D))
 for lbl,f in [('sin',np.sin),('tanh',np.tanh),('abs',np.abs),('cos',np.cos)]: print(lbl,met(np.c_[a(x),f(x)]))
 print('linear',fit(a(x),y)); print('y quant',np.quantile(y,[0,.25,.5,.75,1]))
