from probe import *
from itertools import combinations_with_replacement as cr
print('state singular values',np.linalg.svd(T,compute_uv=False),'telemetry',s['telemetry_vec'])
for m in range(5):print('cal',m,np.round(calfit(m)[1][[20,35,40,42,44]],6))
u,v=s['panel_a'].values(); _,e=calfit(4); f,_=calfit(4); k=np.linalg.norm(f(u)-v,axis=1)<.001; print('cal coefficients',ls(cald(u,4)[0][k],v[k]).round(6))
z=np.array([least_squares(lambda a:f(a[None])[0]-r,r).x for r in x]); np.save('/tmp/z.npy',z)
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[pn]['u'],s[pn][vn]; print('\n',pn,vn)
 for name,terms in [('quadratic', [()]+list(cr(range(4),2))),('fullquadratic',[()]+[(i,) for i in range(4)]+list(cr(range(4),2))),('odd full cubic',[()]+[(i,) for i in range(4)]+list(cr(range(4),3))),('odd separable fifth',[()]+[(i,)*j for j in (1,3,5) for i in range(4)])]:
  d=np.column_stack([np.prod(u[:,j],axis=1) for j in terms]); c=ls(d,v); h=(d@np.linalg.pinv(d)).diagonal();print(name,'rms',np.mean((d@c-v)**2)**.5,'loo',np.mean(((d@c-v)/(1-h[:,None]))**2)**.5);print([(j,row.round(4).tolist()) for j,row in zip(terms,c) if np.max(abs(row))>.03] if name=='fullquadratic' and vn=='v1' else '')
for m in range(1,6):
 d=np.column_stack([np.prod(z[:,j],axis=1) for q in range(m+1) for j in cr(range(4),q)]); p=d[52:]@ls(d[:52],np.c_[Y,T][:52]); print('raw z poly',m,np.mean(abs(p-np.c_[Y,T][52:]),axis=0).round(4))
