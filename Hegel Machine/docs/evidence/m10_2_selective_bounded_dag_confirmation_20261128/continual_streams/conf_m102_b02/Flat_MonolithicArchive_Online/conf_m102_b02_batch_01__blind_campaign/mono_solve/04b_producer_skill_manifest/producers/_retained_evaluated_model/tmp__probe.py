import pickle,numpy as n
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n.set_printoptions(precision=5,suppress=True,linewidth=140)
def mat(x,d):
 return n.column_stack([n.ones(len(x))]+[n.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
def fit(x,y):return n.linalg.lstsq(x,y,rcond=None)[0]
print('times',s['eval_times'],'telemetry',s['telemetry_vec'])
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; x=p['u']; print(k)
 for j,y in p.items():
  if j=='u':continue
  print(j,'range',y.min(0),y.max(0)); A=mat(x,1); b=fit(A,y); print('affine',b); print('affine residual norms',n.sort(n.linalg.norm(A@b-y,axis=1)))
  for transf in (lambda y:y,n.arctanh,n.arcsinh,n.arcsin):
   if transf in (n.arctanh,n.arcsin) and n.max(n.abs(y))>=1:continue
   t=transf(y); print(getattr(transf,'__name__'),*[n.round(n.sqrt(n.mean((mat(x,d)@fit(mat(x,d),t)-t)**2,axis=0)),5) for d in (1,2)])
p=s['train']; x=p['x']; Y=n.c_[p['y'],p['traj']]; print('affine targets',fit(mat(x,1),Y));print('sv vectors',n.linalg.svd(p['traj'],full_matrices=False)[2][:2]);
a=s['panel_a']; A=mat(a['u'],1); keep=n.ones(len(A),bool)
for _ in range(8):
 b=fit(A[keep],a['v'][keep]); r=n.linalg.norm(A@b-a['v'],axis=1); keep=r<max(.001,3*n.median(r))
print('robust A',b,'sv',n.linalg.svd(b[1:],compute_uv=False),'kept',keep.sum()); print('x raw/inverse ranges',x.min(0),x.max(0)); z=n.linalg.solve(b[1:].T,(x-b[0]).T).T;print(z.min(0),z.max(0));print('aligned linear targets',fit(mat(z,1),Y)[:,[0,1,-1]])
for d in (1,2,3,4):
 A=mat(z,d); yp=A[52:]@fit(A[:52],Y[:52]); print('degree',d,n.mean(abs(yp-Y[52:]),axis=0))
