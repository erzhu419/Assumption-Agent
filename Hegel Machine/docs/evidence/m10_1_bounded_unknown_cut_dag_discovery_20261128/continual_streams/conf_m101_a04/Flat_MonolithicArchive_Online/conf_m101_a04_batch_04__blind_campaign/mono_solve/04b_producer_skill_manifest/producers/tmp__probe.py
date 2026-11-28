import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
p=s['panel_a'];u,v=p['u'],p['v'];base=np.ones(len(u));nx=np.roll(u,-1,axis=1)
keep=np.ones(len(u),bool)
for i in range(8):
 f=np.stack([np.ones_like(u),u,u**3,(u-1)*nx],axis=-1); h=np.column_stack([f[:,j]@fit(f[keep,j],v[keep,j]) for j in range(4)]);r=np.linalg.norm(v-h,axis=1);keep=r<max(.001,5*np.median(r))
print('cal cubic',keep.sum(),np.round(np.quantile(r[keep],[0,.5,1]),6));print('cal coeff',np.array([fit(f[keep,j],v[keep,j]) for j in range(4)]))
for tag in ['panel_b','panel_c']:
 u=s[tag]['u']; base=np.ones(len(u));
 for name,v in s[tag].items():
  if name=='u':continue
  def check(parts):
   f=np.column_stack(parts);return '%.5g'%np.sqrt(np.mean((f@fit(f,v)-v)**2))
  print(tag,name,[(k,check([base,u]+z)) for k,z in [('linear',[]),('cube',[u**3]),('square',[u**2]),('poly3',[u**2,u**3]),('full2',[[u[:,i]*u[:,j] for i,j in []],np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)])][1:]),('sin31',[np.sin(3.1*u),np.sin(6.2*u)]),('cos',[np.cos(1.1*u),np.cos(3.9*u)]),('ratSin',[u/(1+.8*abs(u)),np.sin(4.5*u)])]])
t=s['eval_times']; z=s['train']['traj']; print('svd',np.linalg.svd(z,full_matrices=False)[1]); print('svd centered',np.linalg.svd(z-z.mean(0),full_matrices=False)[1]);print('y-tr corr',np.corrcoef(np.c_[s['train']['y'],z].T)[0]);
for family in range(4):
 def tb(p):
  e=-np.expm1(-p[0]*t); b=[t/(1+p[1]*t),-np.expm1(-p[1]*t),np.sin(p[1]*t),t*np.exp(-p[1]*t)][family];return np.array([e,b]).T
 def res(p):
  a=tb(p);return (a@fit(a,z.T)-z.T).ravel()
 q=least_squares(res,[1.5,.5],bounds=([.01,.001],[10,5]));print('time',family,q.x,np.sqrt(np.mean(q.fun**2)))
