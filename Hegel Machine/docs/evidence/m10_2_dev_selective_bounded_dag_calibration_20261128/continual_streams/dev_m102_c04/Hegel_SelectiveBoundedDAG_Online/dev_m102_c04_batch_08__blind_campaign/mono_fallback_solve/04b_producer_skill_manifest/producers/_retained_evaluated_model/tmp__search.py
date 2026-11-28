import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
pa=s['panel_a']; u,v=pa['u'],pa['v']; m=np.linalg.norm(u-v,axis=1)<1; w=np.array([ls(aff(u[m])[:,[0,j+1]],v[m,j]) for j in range(4)]).T
pb=s['panel_b']; wb=ls(aff(pb['u']),pb['v']); B=lambda x:aff(x)@wb
pairs=[(0,1),(0,3),(1,2),(2,3)]; prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cos=lambda x:np.cos(2.4*x)
pc=s['panel_c']; wc1=ls(aff(prod(pc['u'])),pc['v1']); wc2=ls(aff(cos(pc['u'])),pc['v2']); C=lambda x:aff(prod(x))@wc1; D=lambda x:aff(cos(x))@wc2
x=s['train']['x']; z=x*w[1]+w[0]; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; tr=Y[:,1:];
def temporal(p):
 return -np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(temporal(p)@ls(temporal(p),tr[:52].T)-tr[:52].T).ravel(),[.5,2.5],bounds=(.01,30)); print('best rates',o.x); print('B',wb,'C',wc1,'D',wc2,sep='\n');
base={'x':x,'z':z,'inv':(x-w[0])/w[1]}; sets=dict(base)
for nm,a in base.items():
 for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)
for nm,a in list(sets.items()):
 if len(nm)==2 and nm[0] in 'BCD':
  for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)

from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import cho_factor,cho_solve
np.set_printoptions(precision=4,suppress=True)
print('times',t); print('test radii',np.quantile(np.linalg.norm(s['test']['x'],axis=1),[0,.5,1])); print('train radii',np.quantile(np.linalg.norm(x,axis=1),[0,.5,1]));
for k in ['x','z','inv']:
 a=sets[k]; sets['BI'+k]=(a-wb[0])@np.linalg.inv(wb[1:]); sets['BT'+k]=a@wb[1:].T+wb[0]
 for j in ['BI','BT']:
  sets['C'+j+k]=C(sets[j+k]); sets['D'+j+k]=D(sets[j+k]);
for k in ["x","z","inv"]:
 sets["CB"+k]=C(B(sets[k])); sets["DB"+k]=D(B(sets[k]))
base=['x','z','inv','Bx','Bz','Binv','BIx','BIz','BIinv','BTx','BTz','BTinv']; extra={}
for a in base:
 h=sets[a];
 for k in [1.25,2.4]:
  for nm,g in [('cos',np.cos(k*h)),('sin',np.sin(k*h)),('trig',np.c_[np.cos(k*h),np.sin(k*h)])]:extra[f'{nm}{k}{a}']=g
for a in sets:
 if a.startswith(('D','C')):extra[a]=sets[a]
P={d:PolynomialFeatures(d).fit_transform(x) for d in [1,2,3,4]}
train=np.arange(52); test=np.arange(52,72)
U,S,V=np.linalg.svd(tr[:52],full_matrices=False); target=np.c_[Y[:,0],tr@V[:3].T]; scales=target[:52].std(0); target=target/scales

def normalize(F):
 return (F-F[:52].mean(0))/np.maximum(F[:52].std(0),1e-10)
def lin(F):
 F=normalize(F); F=np.c_[np.ones(72),F]; u,sv,v=np.linalg.svd(F[:52],full_matrices=False); uy=u.T@target[:52]; vpred=F@v.T; res=[]
 for reg in [1e-7,.001,.04,.25]:
  h=sv**2/(sv**2+reg); pr=vpred@((sv/(sv**2+reg))[:,None]*uy); loo=(pr[:52]-target[:52])/np.maximum(1e-7,1-(u*u)@h)[:,None]; e=(pr[52:]-target[52:])*scales; res.append((np.mean(abs(loo[:,0])),np.mean(abs((loo[:,1:]*scales[1:])@V[:3])),np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@V[:3])),reg))
 return res

def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(72,-1)
R=[]
def add(n,f):
 for row in lin(f):R.append((*row,n))
for d in [2,3,4]:
 add('poly'+str(d),P[d]);
 for n,g in extra.items():
  if d<4:
   add(f'p{d}+{n}',np.c_[P[d],g]); add(f'p{d}+x*{n}',np.c_[P[d],g,cross(x,g)]);
  if d==2:
   for k in ['Cx','Cz','Cinv','CBx','CBz','CBinv','CBIx','CBIz','CBIinv']:
    a=sets[k]; add(f'p{d}+{k}*{n}',np.c_[P[d],g,cross(a,g)]); add(f'p{d}+x*{k}+{n}',np.c_[P[d],g,cross(a,x)]); add(f'p{d}+x*{k}+x*{n}',np.c_[P[d],g,cross(a,x),cross(x,g)])
for j in range(4): print('leader',j,[(np.round(r[:5],5),r[-1]) for r in sorted(R,key=lambda r:r[j])[:8]])
