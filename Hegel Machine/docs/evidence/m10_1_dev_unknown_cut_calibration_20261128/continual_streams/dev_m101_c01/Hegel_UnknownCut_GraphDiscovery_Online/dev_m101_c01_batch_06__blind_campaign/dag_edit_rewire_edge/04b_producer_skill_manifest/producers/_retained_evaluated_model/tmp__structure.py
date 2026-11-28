exec(open('/tmp/readout_search.py').read().split('c=s[')[0])
from scipy.linalg import lu_factor,lu_solve
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
z=lin(x)@coef; inv=(x-coef[0])@np.linalg.inv(coef[1:]); Z={'x':x,'B+':z,'B-':inv,'raw':raw}; p=s['panel_c']; u=p['u']
funcs={'tanh':np.tanh,'sin':np.sin,'erf':erf,'atan':np.arctan,'sqrt':lambda a:a/(1+a*a)**.5,'rat':lambda a:a/(1+a*a),'log':lambda a:np.arcsinh(a),'gauss':lambda a:1-np.exp(-a*a)}
def cv(F,O):
 P=np.linalg.pinv(F,rcond=1e-9); h=(F*P.T).sum(1); return np.mean(np.abs((O-F@(P@O))/(1-h[:,None])))
def pfit(F,O):
 return np.linalg.lstsq(F,O,rcond=1e-9)[0]
for key in ('v1','v2'):
 O=p[key]; records=[]
 for name,f in funcs.items():
  for kind in ('only','raw','interaction','addquad','shortquad'):
   def design(v,k):
    h=f(k*v)
    return {'only':lin(h),'raw':np.column_stack((lin(v),h)),'interaction':np.column_stack((lin(h),*[h[:,i]*h[:,j] for i in range(4) for j in range(i+1,4)])),'addquad':np.column_stack((quad(v),h)),'shortquad':np.column_stack((lin(v),h,h*h))}[kind]
   grid=np.geomspace(.1,4,30); j=np.argmin([cv(design(u,k),O) for k in grid]); r=minimize_scalar(lambda k:cv(design(u,k),O),bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),method='bounded'); records.append((r.fun,name,kind,r.x,np.mean(np.abs(design(u,r.x)@pfit(design(u,r.x),O)-O))))
 print('panel',key, sorted(records)[:10]); print('quad coefficients',pfit(quad(u),O).round(4))
for k,v in Z.items():
 F=quad(v); beta=pfit(F,Y); print(k,'scalar quad beta',beta[:,0].round(3),'trajectory quad beta',beta[:,-1].round(3))
print('traj rank-one noise per time',np.linalg.svd(Y[:,1:],full_matrices=False)[2][0].round(4)); print('y vs trajectory amplitude',pearsonr(Y[:,0],Y[:,-1]).statistic)
rows=np.arange(n); r=np.linalg.norm(x-x.mean(0),axis=1); rng=np.random.default_rng(42); blocks=[(rows[:52],rows[52:]),*[(p[:52],p[52:]) for p in (rng.permutation(n) for _ in range(10))],(np.argsort(r)[:-12],np.argsort(r)[-12:])]; rec=[]
for k,v in Z.items():
 for transform,f in funcs.items():
  if transform=='gauss':continue
  for scale in (.35,.7,1,1.4,2):
   h=f(scale*v); feats={'nonlinear quad':quad(h),'quadratic plus odd':np.column_stack([quad(v),h]),'quadratic plus odd and even':np.column_stack([quad(v),h,h*h]),'nonlinear odd mixed':np.column_stack([quad(v),h,*[v[:,i]*h[:,j] for i in range(4) for j in range(4)]])}
   for nm,F in feats.items():
    fr=[]
    for tr,te in blocks:
     b=pfit(F[tr],Y[tr]); e=np.abs(F[te]@b-Y[te]); fr.append([e[:,0].mean(),e[:,1:].mean()]);
    er=np.array(fr); rec.append((k,transform,scale,nm,er[0],er[:-1].mean(0),er[:-1].max(0),er[-1]))
for j in (0,1):
 for a in sorted(rec,key=lambda a:a[5][j]+.2*a[7][j])[:8]: print('readout',j,*a[:4],*[np.round(e,5).tolist() for e in a[4:]])
