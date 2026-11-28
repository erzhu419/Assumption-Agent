import pickle,glob,importlib.util,numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement as cwr
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; t=s['eval_times']; Y=np.c_[d['y'],d['traj']]; a=s['panel_a']; u=a['u']; v=a['v']; mods={}
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 n=f.split('_')[-2]; m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('p',f));sp.loader.exec_module(m);mods[n]=m
Z={'raw':x}
for n in ('00','01','03','09'):
 m=mods[n]; c=m._calibration(a);print('cal',n,c)
 if hasattr(m,'_decode'): Z[n]=m._decode(x,c)
 elif n=='00': Z[n]=np.linalg.solve(c[1:].T,(x-c[0]).T).T
 else: Z[n]=(x-c[0])/c[1]
for name in ['panel_b','panel_c']:
 p=s[name]; u=p['u']; print('panel',name)
 for k in p.keys()-{'u'}:
  v=p[k]
  for m,fun in [('linear',lambda a:np.c_[np.ones(len(a)),a]),('warps',lambda a:np.c_[np.ones(len(a)),a,np.tanh(2.15*a),np.sin(1.25*a)]),('quadratic',lambda a:np.c_[np.ones(len(a)),a,*[a[:,i]*a[:,j] for i,j in cwr(range(4),2)]])]:
   X=fun(u); h=X@np.linalg.pinv(X); print(k,m,np.mean(abs((v-h@v)/(1-np.diag(h))[:,None])),np.mean(abs(v-h@v)))
for n,z in Z.items():
 z=z/np.std(z[:52],axis=0); sq=cdist(z,z,'sqeuclidean'); D=np.sqrt(sq); ip=z@z.T; models=[]
 for order in range(1,5):
  terms=np.array([np.prod(z[:,ix],axis=1) for k in range(1,order+1) for ix in cwr(range(4),k)]).T; terms/=np.maximum(terms[:52].std(axis=0),1e-8); models.append(('poly'+str(order),1+terms@terms.T))
 for l in (.5,1,2,4,8):
  r=np.sqrt(5)*D/l; models.extend([(f'mat{l}',(1+r+r*r/3)*np.exp(-r)+.1*ip+1),(f'rbf{l}',np.exp(-sq/(2*l*l))+.1*ip+1)])
 for name,K in models:
  vals=[]
  for lam in (1e-7,1e-4,.003,.03,.3):
   yh=K[52:,:52]@solve(K[:52,:52]+lam*np.eye(52),Y[:52],assume_a='pos'); e=np.mean(abs(yh-Y[52:]),axis=0); vals.append((max(e[0],e[1:].mean()),e[0],e[1:].mean(),e[-1],lam))
  print(n,name,np.round(min(vals),5))
