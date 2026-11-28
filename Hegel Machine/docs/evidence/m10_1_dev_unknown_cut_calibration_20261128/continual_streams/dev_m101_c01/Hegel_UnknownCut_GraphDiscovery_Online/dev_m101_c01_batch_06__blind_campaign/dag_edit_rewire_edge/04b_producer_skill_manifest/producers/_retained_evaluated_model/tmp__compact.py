exec(open('/tmp/readout_search.py').read().split('c=s[')[0])
from scipy.optimize import least_squares
from scipy.linalg import qr
p=s['panel_c']; u=p['u'];
# Profile the shared activation parameters against all four calibration channels.
acts={'asinh':lambda z,b:np.arcsinh(z), 'power':lambda z,b:z*(1+z*z)**b,'softsign':lambda z,b:z/(1+abs(z)**b)**(1/b),'tanh':lambda z,b:np.tanh(z),'sin':lambda z,b:np.sin(z),'atan':lambda z,b:np.arctan(z),'logcosh':lambda z,b:np.logaddexp(z,-z)-np.log(2),'erf':lambda z,b:__import__('scipy').special.erf(z)}
cal={}
for key in ('v1','v2'):
 O=p[key]; rec=[]
 for name,act in acts.items():
  for kind in ('odd','diag','mix'):
   def feat(z,par):
    h=act(np.exp(par[0])*z,par[1]); return np.column_stack([lin(z),h,*([h*h] if kind!='odd' else []),*([z*h] if kind=='mix' else [])])
   def fit(ids,par):
    f=feat(u[ids],par); return np.linalg.lstsq(f,O[ids],rcond=1e-8)[0]
   def fitp(ids):
    return least_squares(lambda par:(feat(u[ids],par)@fit(ids,par)-O[ids]).ravel(),[0.,1.5 if name=='softsign' else -.4],bounds=([-2.5,.5 if name=='softsign' else -3],[2.,4]),max_nfev=55).x
   ids=np.arange(len(u)); par=fitp(ids); ff=feat(u,par); w=fit(ids,par); iv=np.linalg.pinv(ff); cv=np.mean(abs((ff@w-O)/(1-(ff*iv.T).sum(1))[:,None])); rec.append((cv,float(np.mean(abs(ff@w-O))),name,kind,par,w))
 rec.sort(key=lambda a:a[0]); cal[key]=rec[:4]; print(key,[(r[:4],r[4].round(4)) for r in rec[:7]])
 def build(z,rec):
  _,_,name,kind,par,w=rec; h=acts[name](np.exp(par[0])*z,par[1]); return np.column_stack([lin(z),h,*([h*h] if kind!='odd' else []),*([z*h] if kind=='mix' else [])])@w
# Test composed panel readout features and a low-rank trajectory representation.
B=lambda z:lin(z)@coef
Bi=lambda z:(z-coef[0])@np.linalg.inv(coef[1:]); X={'x':x,'B+':B(x),'B-':Bi(x)}; F={'quad':quad(x)}
for name,z in X.items():
 g=build(z,cal['v1'][0]); h=build(z,cal['v2'][0]); F[name+'maps']=np.column_stack([quad(x),g,h]); F[name+'compose']=np.column_stack([quad(x),g,h,build(g,cal['v2'][0])]); F[name+'mapquad']=np.column_stack([quad(g),quad(h)[:,1:]]);
rows=np.arange(n); rng=np.random.default_rng(42); radius=np.linalg.norm(x-x.mean(0),axis=1); edge=np.argsort(radius)[-12:]; splits=[(rows[:52],rows[52:])]+[(p[:52],p[52:]) for p in (rng.permutation(n) for _ in range(10))]+[(np.setdiff1d(rows,edge),edge)]; D=cdist(x,x,'sqeuclidean'); kernels={'matern':banks['base'][2,:n,:n], 'rbf':banks['base'][1,:n,:n], 'broad':np.exp(-D/5)}
recs=[]
for name,f in F.items():
 for ks,K in kernels.items():
  for alpha in (.0003,.001,.004):
   for reg in ([0.] if name=='quad' else [.0001,.01,1]):
    errors=[]
    for tr,te in splits:
     Ft=f[tr]; Q=cho_factor(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr))); Fi=cho_solve(Q,Ft); Yi=cho_solve(Q,Y[tr]); cov=Ft.T@Fi; w=np.linalg.lstsq(cov+reg*np.diag(np.maximum(1.e-9,cov.diagonal())),Ft.T@Yi,rcond=1.e-9)[0]; yhat=f[te]@w+K[np.ix_(te,tr)]@(Yi-Fi@w);
     _,_,vt=np.linalg.svd(Y[tr,1:],full_matrices=False); h=vt[0]; state_rank=(yhat[:,1:]@h)[:,None]*h; err=abs(yhat-Y[te]); errors.append([err[:,0].mean(),err[:,1:].mean(),abs(state_rank-Y[te,1:]).mean()])
    e=np.array(errors); recs.append(((name,ks,alpha,reg),e[0],e[:-1].mean(0),e[:-1].max(0),e[-1]))
for col in (0,1,2):
 print('metric',col)
 for r in sorted(recs,key=lambda r:r[2][col]+.3*r[4][col])[:7]: print(r[0],*[v.round(5).tolist() for v in r[1:]])
print('baseline',*[v.round(5).tolist() for v in next(r for r in recs if r[0]==('quad','matern',.001,0.))[1:]])
