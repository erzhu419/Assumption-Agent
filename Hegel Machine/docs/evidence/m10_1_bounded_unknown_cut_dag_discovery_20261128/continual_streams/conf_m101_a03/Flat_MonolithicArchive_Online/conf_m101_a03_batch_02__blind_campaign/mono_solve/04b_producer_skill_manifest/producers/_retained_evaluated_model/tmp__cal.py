exec(open('/tmp/explore.py').read().split('u,v=(s')[0])
from scipy.optimize import least_squares
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.arange(len(u))
for _ in range(3):
 c=fit(A(u[keep]),v[keep]); keep=np.argsort(np.linalg.norm(v-A(u)@c,axis=1))[:45]
for terms in [[(i,i) for i in range(4)],[(i,i) for i in range(4)]+[(i,i,i) for i in range(4)],[(i,i) for i in range(4)]+[(i,i,i) for i in range(4)]+[(i,(i+1)%4) for i in range(4)],list(combinations_with_replacement(range(4),2))+list(combinations_with_replacement(range(4),3))]:
 def f(w): return np.column_stack([A(w)]+[np.prod(w[:,t],axis=1) for t in terms])
 coef=fit(f(u[keep]),v[keep]); print('terms',terms,'rms',np.std(f(u[keep])@coef-v[keep])); print(coef)
 if len(terms)==8: base=coef.copy(); base_terms=terms.copy()
print('u ranges',u.min(0),u.max(0));print('test ranges',s['test']['x'].min(0),s['test']['x'].max(0));
for ex in [[lambda w:w*w,lambda w:w**3],[lambda w:w*w,lambda w:np.tanh(w)],[lambda w:w*w,lambda w:np.sin(w)],[lambda w:w*w,lambda w:w**3,lambda w:np.sin(w)]]:
 def f(w): return np.column_stack([A(w)]+[g(w) for g in ex])
 coef=fit(f(u[keep]),v[keep]); print('alt rms',np.std(f(u[keep])@coef-v[keep]));print(coef)
