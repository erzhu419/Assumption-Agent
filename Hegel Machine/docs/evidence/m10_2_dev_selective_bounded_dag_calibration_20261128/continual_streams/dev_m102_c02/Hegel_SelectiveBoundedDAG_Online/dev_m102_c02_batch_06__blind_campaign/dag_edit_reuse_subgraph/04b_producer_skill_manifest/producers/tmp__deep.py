import pickle,numpy as np,importlib.util
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from scipy.linalg import subspace_angles

def load(path):
 sp=importlib.util.spec_from_file_location('m',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
n=load('/holdout_task/node_00.py'); f=load('/holdout_task/_frozen_nodes/node_01.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; z=n._invert(x,n._fit_sensor(s['panel_a'])); np.set_printoptions(precision=4,suppress=True); print('first z and targets',np.c_[z,y,t[:,-1]][:10]); print('study keys',list(s));
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def rat(x,p):return x/(1+np.exp(p)*x*x)
def poly(x,j):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ix],axis=1) for k in range(1,j+1) for ix in combinations_with_replacement(range(4),k)])
def evaluate(q):
 vals=[]
 for j,i in [(np.arange(52),np.arange(52,72)),(np.arange(18,72),np.arange(18)),(np.r_[0:18,36:72],np.arange(18,36))]:
  vals.append([np.mean(abs(f._scalar_readout(q[j],q[i],y[j])-y[i])),np.mean(abs(f._trajectory(q[j],q[i],t[j],s['eval_times'])-t[i]))]);
 return np.array(vals).round(4).tolist()
print('poly predictions');
for base in [z,x]:
 for k in range(1,5):
  d=poly(base,k); r=np.c_[y,t]-d@fit(d,np.c_[y,t]); print(k,np.std(r,axis=0)[[0,1,-1]],np.mean(abs(np.c_[y,t][52:]-d[52:]@fit(d[:52],np.c_[y,t][:52])),axis=0)[[0,1,-1]])
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][key]; print(panel,key,'baseline',evaluate(z));
 for U,V in [(u,v),(v,u)]:
  sc,W=n._fit_coordinates({'u':U,'v':V}); print('rat affine',sc,'err',np.std(n._design(U,sc)@W-V),evaluate(n._design(z,sc)@W)); print('rat coefficients',W)
  for k in [2,3]: print('poly',k,evaluate(poly(z,k)@fit(poly(U,k),V)))
 # Linear mixing before a monotone coordinatewise squashing, variable projection.
 for form in ['out_sqrt','out_cubic','out_atanh','out_plusrat','in_mixrat','in_mixrat_shared']:
  if form.startswith('out_'):
   def target(p):
    ctr=p[:4]; d=V0-ctr; h=np.exp(p[4:8]);
    if form=='out_sqrt': return d/(np.sqrt(np.maximum(1-h*d*d,1e-6)))
    if form=='out_cubic':return d+h*d**3
    if form=='out_atanh':return np.arctanh(np.clip(d*h,-.9999,.9999))/h
    return d+p[8:]*rat(d,p[4:8]);
   V0=v; A=aff(u); P=np.linalg.pinv(A); ini=np.r_[np.zeros(4),np.full(4,-2.),np.ones(4)] if form=='out_plusrat' else np.r_[np.zeros(4),np.full(4,-2.)];
   try:
    o=least_squares(lambda p:(A@(P@target(p))-target(p)).ravel(),ini,bounds=(-4,4),max_nfev=120); print(form,np.std(o.fun))
   except Exception as e:print(e)
  else:
   # Sum of four output-specific ridge terms added to affine output.
   A=aff(u); P=np.linalg.pinv(A); out=[]; q=[]
   for j in range(4):
    def D(U,w):
     h=U@w[:4]+w[4];return np.c_[aff(U),h/(1+h*h)]
    opts=[]
    for k in range(4):
     w=np.zeros(5);w[k]=.8
     def r(w):
      d=D(u,w);return d@fit(d,v[:,j])-v[:,j]
     opts.append(least_squares(r,w,bounds=(-4,4),max_nfev=110));
    o=min(opts,key=lambda a:a.cost);out.append(np.std(o.fun));q.append(D(z,o.x)@fit(D(u,o.x),v[:,j]));
   print(form,out,evaluate(np.array(q).T)); break
