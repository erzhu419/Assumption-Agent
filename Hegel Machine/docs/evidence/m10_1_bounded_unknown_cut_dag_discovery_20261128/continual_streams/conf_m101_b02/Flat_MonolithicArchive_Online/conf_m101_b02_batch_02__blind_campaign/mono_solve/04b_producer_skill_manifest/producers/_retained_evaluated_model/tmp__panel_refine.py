exec(open('/tmp/audit_batch.py').read().split('for name,a,b in folds:')[0])
for name in ['panel_a','panel_b']:
 p=s[name]; d=affine(p['u']); coef=c if name=='panel_a' else b; e=np.linalg.norm(d@coef-p['v'],axis=1)
 print(name,'residual quantiles',np.quantile(e,[0,.25,.5,.75,.85,.9,1]))
u=forward_b(inverse_a(x)); test=forward_b(inverse_a(s['test']['x'])); print('latent ranges',u.min(0),u.max(0),test.min(0),test.max(0))
terms=[k for degree in range(4) for k in combinations_with_replacement(range(4),degree)]
g=np.column_stack([u[:,k].prod(1) for k in terms]); coef=ls(g,z)
print('latent term coefficients for scalar and final state')
for k,cy,ct in zip(terms,coef[:,0],coef[:,-1]): print(k,round(cy,5),round(ct,5))
sel=[j for j,k in enumerate(terms) if len(k)<3 or any(set(p).issubset(k) for p in support)]; other=np.setdiff1d(range(len(terms)),sel)
def blend(a,b,pen):
 if pen==0: return pred(g,a,b)
 target,v=coordinates(a,'svd'); main=g[:,sel]; extra=g[:,other]; proj=ls(main[a],extra[a]); q=extra-main@proj
 w=np.linalg.norm(q[a],axis=0).clip(1e-12); q=q/w
 mat=np.c_[main[a],q[a]]; reg=np.c_[np.zeros((len(other),len(sel))),np.sqrt(pen)*np.eye(len(other))]
 h=np.c_[main[b],q[b]]@ls(np.r_[mat,reg],np.r_[target,np.zeros((len(other),3))]); return np.c_[h[:,0],h[:,1:]@v]
for name,a,b in folds:
 print(name,'panel only',metric(pred(g[:,sel],a,b),b).round(5),'shrink',[(p,metric(blend(a,b,p),b).round(5).tolist()) for p in [.1,1,10]])
