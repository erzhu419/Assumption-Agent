import pickle,numpy as np,importlib.util
from itertools import product
from scipy.optimize import least_squares
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
n=load('n','/holdout_task/node_00.py'); f=load('f','/holdout_task/_frozen_nodes/node_01.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; coeff=n._fit_sensor(s['panel_a']);z=n._invert(x,coeff); a,g,h,p,c=coeff;b=np.roll(z,-1,axis=1); print('inverse ranges',z.min(0),z.max(0),'residual max',np.max(abs(a+g*z+h*b+p*z*b+c*z**3-x)));print('panel ranges',[(k,[(l,round(np.std(v),3)) for l,v in s[k].items()]) for k in ['panel_a','panel_b','panel_c']]);
def design(z,mode,scale):
 if mode=='aff':return np.c_[np.ones(len(z)),z]
 if mode=='sq':return np.c_[np.ones(len(z)),z,z*z]
 return n._design(z,scale)
M={}
for label,panel,key,mode in [('B','panel_b','v','rat'),('b','panel_b','v','aff'),('C','panel_c','v1','rat'),('c','panel_c','v1','aff'),('D','panel_c','v2','sq'),('d','panel_c','v2','aff')]:
 pn=s[panel]; u,v=pn['u'],pn[key]; scale=np.zeros(4)
 if mode=='rat':scale,w=n._fit_coordinates({'u':u,'v':v})
 else:w=np.linalg.lstsq(design(u,mode,scale),v,rcond=None)[0]
 M[label]=lambda z,mode=mode,scale=scale,w=w:design(z,mode,scale)@w
rng=np.random.default_rng(47); splits=[np.array_split(np.arange(len(x)),4)[i] for i in range(4)] + [a for a in np.array_split(rng.permutation(len(x)),4)]; opts=[]
for root in 'zx':
 for seq in ['']+list(M)+[a+b for a,b in product(M,M) if a.lower()!=b.lower()]+[a+b+d for a,b,d in product('BC','BCD','BCD') if len({a,b,d})==3]+['CDB','DBC','DCB','BCD']:
  q=z if root=='z' else x
  for k in seq:q=M[k](q)
  es=[]
  for i in splits:
   j=np.setdiff1d(np.arange(len(x)),i); yh=f._scalar_readout(q[j],q[i],y[j]); th=f._trajectory(q[j],q[i],t[j],s['eval_times']);es.append([np.mean(abs(yh-y[i])),np.mean(abs(th-t[i]))]);
  es=np.array(es); opts.append((np.max(es.mean(0)),root+seq,es.mean(0).round(5).tolist(),es.max(0).round(5).tolist()))
print('BEST',*sorted(opts)[:25],sep='\n');print('SINGLES',*[a for a in opts if len(a[1])<=2],sep='\n')
