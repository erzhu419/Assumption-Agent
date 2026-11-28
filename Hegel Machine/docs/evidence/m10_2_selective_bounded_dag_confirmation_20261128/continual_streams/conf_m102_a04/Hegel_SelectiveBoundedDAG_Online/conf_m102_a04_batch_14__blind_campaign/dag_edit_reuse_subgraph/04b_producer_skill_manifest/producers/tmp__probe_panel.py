import numpy as np,pickle
from scipy.optimize import minimize_scalar, least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
def feat(*args):return np.column_stack((np.ones(len(args[0])),*args))
u,v=(s['panel_a'][k] for k in ('u','v')); n=np.roll(u,-1,axis=1); p=np.roll(u,1,axis=1)
for j in range(4):
 a=u[:,j]; b=v[:,j]; t=n[:,j]; r=p[:,j]; print('coord',j)
 for key,f in [('linear',feat(a)),('archfixed',feat(a)),('archfree',feat(a,t*(a-1),a**3)),('cyclic',feat(a,t,r,a*t,a*r,a**2,a**3,t**2,r**2)),('full',feat(u,*[u[:,i]*u[:,k] for i in range(4) for k in range(i,4)],u**3))]:
  y=b-(.32*t*(a-1)+.16*a**3) if key=='archfixed' else b
  q=least_squares(lambda q:f@q-y,ls(f,y),loss='cauchy',f_scale=.002).x
  for _ in range(3):
   e=np.abs(f@q-y); keep=e<max(5*np.median(e),.0005); q=ls(f[keep],y[keep])
  e=np.abs(f@q-y);print(key,np.round(q,5),np.round(np.quantile(e,[.5,.75,.85,.95,1]),6))
u=s['panel_c']['u']; funcs={
 'softsign':lambda a,q:feat(a/(1+q*abs(a)),a), 'curvature':lambda a,q:feat(a,a*a,a*a/(1+q*a*a)),
 'atan':lambda a,q:feat(a,a**3,np.arctan(q*a)), 'atanlin':lambda a,q:feat(a,np.arctan(q*a)), 'cos':lambda a,q:feat(np.cos(q*a)), 'cosquad':lambda a,q:feat(a,a*a,np.cos(q*a)), 'sin':lambda a,q:feat(np.sin(q*a)), 'sinlin':lambda a,q:feat(a,np.sin(q*a)), 'tanh':lambda a,q:feat(np.tanh(q*a),a), 'softsat':lambda a,q:feat(a/np.sqrt(1+q*a*a),a), 'ratodd':lambda a,q:feat(a/(1+q*a*a),a), 'exp':lambda a,q:feat(np.exp(-q*a*a)), 'coslinear':lambda a,q:feat(a,np.cos(q*a)), 'cos_sin':lambda a,q:feat(np.cos(q*a),np.sin(q*a))}
for name in ('v1','v2'):
 y=s['panel_c'][name];print(name)
 for key,fun in funcs.items():
  def score(q):
   f=fun(u,q);return np.linalg.norm(f@ls(f,y)-y)
  g=np.geomspace(.04,10,50); i=np.argmin([score(q) for q in g]); q=minimize_scalar(score,bounds=(g[max(i-1,0)],g[min(i+1,len(g)-1)]),method='bounded').x
  f=fun(u,q); h=f@np.linalg.pinv(f); e=(y-h@y)/np.maximum(1-h.diagonal()[:,None],1e-8)
  print(key,round(q,5),round(score(q),6),round(np.mean(np.abs(e)),6))
