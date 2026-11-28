import sys,pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task');import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; b=s['panel_b']; w=m._ls(np.c_[np.ones(len(b['u'])),b['u']],b['v']);print('B',w,'orth',w[1:]@w[1:].T); print('C grid',np.round(x[:12],3));
forms={'abs':lambda q,k:q/(1+k*abs(q)),'sqden':lambda q,k:q/(1+k*q*q),'even':lambda q,k:q*q/(1+k*q*q),'tanh':lambda q,k:np.tanh(k*q),'sin':lambda q,k:np.sin(k*q),'poly3':lambda q,k:q**3,'exp':lambda q,k:np.exp(k*q),'sinh':lambda q,k:np.sinh(k*q)}
for key in ('v1','v2'):
 y=s['panel_c'][key]; out=[]
 for bn,z in [('raw',x),('rot',np.c_[np.ones(len(x)),x]@w),('inv',(x-w[0])@np.linalg.pinv(w[1:]))]:
  ix=list(itertools.combinations(range(4),2)); prod=np.column_stack([z[:,i]*z[:,j] for i,j in ix]); plus=np.column_stack([z[:,i]+z[:,j] for i,j in ix]); minus=np.column_stack([z[:,i]-z[:,j] for i,j in ix]); ring=prod[:,[0,3,5,2]]; opp=prod[:,[1,4]]
  for base,bf in [('a',np.c_[np.ones(len(x)),z]),('ring',np.c_[np.ones(len(x)),z,ring]),('cross',np.c_[np.ones(len(x)),z,prod]),('quad',np.c_[np.ones(len(x)),z,z*z,prod])]:
   for name,f in forms.items():
    for zn,zz in [('z',z),('prod',prod),('ring',ring),('plus',plus),('minus',minus)]:
     def loss(k):
      d=np.c_[bf,f(zz,k)]; return np.mean((d@m._ls(d,y)-y)**2)**.5
     opt=minimize_scalar(loss,bounds=(.01,4),method='bounded');out.append((opt.fun,bn,base,name,zn,opt.x))
 print(key,sorted(out)[:22]); print('quadratic coeff',np.round(m._ls(m._design(x,m._terms(x,2)),y),3))
