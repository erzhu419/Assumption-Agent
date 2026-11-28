import pickle,numpy as np
from scipy.linalg import lstsq
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=(s['train'][k] for k in ('x','y','traj'))
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aff(a):return np.c_[np.ones(len(a)),a]
def poly(a,d):return np.column_stack([np.prod(a[:,ix],axis=1) for j in range(d+1) for ix in combinations_with_replacement(range(a.shape[1]),j)])
def report(a,b):
 z=lstsq(a[:52],b[:52])[0];r=a@z-b;return np.mean(abs(r[:52])),np.mean(abs(r[52:]))
p=s['panel_a']; u,v=p['u'],p['v']; off=[]; gain=[]
for j in range(4):
 a,b=u[:,j],v[:,j]; ii,jj=np.triu_indices(len(a),1); dx=a[jj]-a[ii]; ok=abs(dx)>.1*np.ptp(a); g=np.median((b[jj[ok]]-b[ii[ok]])/dx[ok]); o=np.median(b-g*a); r=b-o-g*a; k=abs(r)<6*np.median(abs(r))+1e-10; o,g=lstsq(aff(a[k,None]),b[k])[0];off.append(o);gain.append(g);print('cal',j,o,g,k.sum(),np.std((b-o-g*a)[k]))
z=(x-off)/gain
ops={}
for pn in ['panel_b','panel_c']:
 p=s[pn];u=p['u'];
 for k,v in p.items():
  if k=='u':continue
  op=lstsq(aff(u),v)[0];ops[pn[-1]+k[-1]]=op
  print(pn,k,'op',op,'residuals',[np.std(poly(u,d)@lstsq(poly(u,d),v)[0]-v) for d in (1,2)],'linear_svs',np.linalg.svd(op[1:])[1])
print('times',s['eval_times']);print('traj singulars',np.linalg.svd(t,compute_uv=False));
for j in range(1,5):print('poly',j,report(poly(z,j),y),report(poly(z,j),t))
P=poly(z,2); beta=lstsq(P[:52],y[:52])[0]; print('quad beta',beta,'linear traj',lstsq(aff(z),t)[0]);print('worst quadratic residual',np.sort(abs(P@beta-y))[-12:]);print('largest y residual rows',np.c_[z,y,P@beta-y][np.argsort(abs(P@beta-y))[-12:]])
for name,op in ops.items():
 c=aff(z)@op;print(name, 'quad coef in panel coords',lstsq(poly(c,2),y)[0], 'bilinear',report(np.c_[aff(c),np.sum(c[:,:2]*c[:,2:],axis=1)],y))
H=np.zeros((4,4)); ij=np.triu_indices(4);H[ij]=beta[5:];H=(H+H.T)/2; print('quad eig',np.linalg.eigh(H));
