from profile_spatial import *
from scipy.optimize import least_squares, root
n=len(s['train']['x']); x=np.r_[s['train']['x'],s['test']['x']]; u,v=[s['panel_a'][k] for k in ('u','v')]; k=2.15
D=lambda x:aff(np.c_[x,np.sin(k*x)]); a=D(u); w=ls(a,v)
for _ in range(10):
 e=np.linalg.norm(a@w-v,axis=1); mask=e<=4*np.median(e); w=ls(a[mask],v[mask]);
print('cal errors',np.round(np.sort(e),5)); print('cal coef',np.round(w,5))
a0=ls(aff(u[mask]),v[mask]); seeds=np.linalg.solve(a0[1:].T,(x-a0[0]).T).T
q=np.array([root(lambda z:D(z[None,:])[0]@w-row,seed,tol=1e-10).x for row,seed in zip(x,seeds)]); print('inversion',np.max(np.abs(D(q)@w-x)),np.min(q,axis=0),np.max(q,axis=0))
u=s['panel_c']['u']; P=lambda a:aff(prod(a,[(0,1),(0,3),(1,2),(2,3)])); c=P(q)@ls(P(u),s['panel_c']['v1']); C=lambda a:aff(np.cos(2.4*a)); d=C(q)@ls(C(u),s['panel_c']['v2']); print('panelC coef',np.round(ls(P(u),s['panel_c']['v1']),4),np.round(ls(C(u),s['panel_c']['v2']),4))
cross=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
aff0=aff(np.c_[q,c,d]); yc=np.c_[s['train']['y'],s['train']['traj']]; blocks={'q*c':cross(q,c),'c*d':cross(c,d),'q*d':cross(q,d),'q*q':cross(q,q),'c*c':cross(c,c),'d*d':cross(d,d),'q*c diag':q*c,'c*d diag':c*d,'q*d diag':q*d}; blocks.update({k+' trace':v.sum(axis=1,keepdims=True) for k,v in list(blocks.items()) if 'diag' in k});
for bn,adds in [('base',[]),('arch',['q*c','c*d']),('alltraces',[k for k in blocks if 'trace' in k])]+[(k,[k]) for k in blocks]:
 f=np.column_stack([aff0]+[blocks[k] for k in adds]); pred=f[52:n]@ls(f[:52],yc[:52]); et=np.abs(pred-yc[52:]); ins=np.abs(f[:52]@ls(f[:52],yc[:52])-yc[:52]); print(bn,np.linalg.matrix_rank(f[:52]),np.round([et[:,0].mean(),et[:,1:].mean(),ins[:,0].mean(),ins[:,1:].mean()],6))
