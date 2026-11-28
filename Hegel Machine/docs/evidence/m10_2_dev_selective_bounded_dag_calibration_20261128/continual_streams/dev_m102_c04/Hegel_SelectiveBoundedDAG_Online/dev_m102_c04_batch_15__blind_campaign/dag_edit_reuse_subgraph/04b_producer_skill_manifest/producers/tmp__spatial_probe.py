import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tri=list(itertools.combinations(range(4),2)); tri3=list(itertools.combinations(range(4),3))
def affine(u):return np.c_[np.ones(len(u)),u]
def fit(a,v):return np.linalg.lstsq(a,v,rcond=1e-10)[0]
def scan(u,v,fn,lo,hi):
 def obj(k):
  a=np.c_[affine(u),fn(u,k)];return np.mean((a@fit(a,v)-v)**2)
 ks=np.linspace(lo,hi,81);j=int(np.argmin([obj(k) for k in ks]));m=minimize_scalar(obj,bounds=(ks[max(0,j-1)],ks[min(80,j+1)]),method='bounded');return (np.sqrt(m.fun),m.x)
def pair(u):return np.column_stack([u[:,i]*u[:,j] for i,j in tri])
def trip(u):return np.column_stack([u[:,i]*u[:,j]*u[:,l] for i,j,l in tri3])
pa=s['panel_a'];m=np.linalg.norm(pa['v']-pa['u'],axis=1)<4
print('A sine',scan(pa['u'][m],pa['v'][m],lambda x,k:np.sin(k*x),.1,8),'affine',np.std(affine(pa['u'][m])@fit(affine(pa['u'][m]),pa['v'][m])-pa['v'][m]))
u=s['panel_c']['u'];v=s['panel_c']['v2'];b=s['panel_b'];bw=fit(affine(b['u']),b['v']);print('B affine',bw);q=affine(u)@bw
for tag,x in [('raw',u),('B direct',q),('B inverse',(u-bw[0])@np.linalg.inv(bw[1:]))]:
 print(tag)
 for n,fun in [('square',lambda x,k:x*x/(1+k*x*x)),('pair',lambda x,k:pair(x)/(1+k*pair(x)**2)),('pair2',lambda x,k:pair(x)/(1+k*abs(pair(x)))),('trip',lambda x,k:trip(x)/(1+k*trip(x)**2)),('odd',lambda x,k:x/(1+k*x*x)),('tanh',lambda x,k:np.tanh(k*x)),('sin',lambda x,k:np.sin(k*x)),('cos',lambda x,k:np.cos(k*x)),('quadratic',lambda x,k:np.c_[x*x,pair(x)]),('squares+rational pair',lambda x,k:np.c_[x*x,pair(x)/(1+k*pair(x)**2)]),('squares+cube',lambda x,k:np.c_[x*x,x**3]),('square+sine',lambda x,k:np.c_[x*x,np.sin(k*x)]),('pairwise distances',lambda x,k:np.column_stack([(x[:,i]-x[:,j])**2/(1+k*(x[:,i]-x[:,j])**2) for i,j in tri]))]:
  print(n,scan(x,v,fun,0.001,6))
print('affine V2 residual singular values',np.linalg.svd(v-affine(u)@fit(affine(u),v),compute_uv=False))
