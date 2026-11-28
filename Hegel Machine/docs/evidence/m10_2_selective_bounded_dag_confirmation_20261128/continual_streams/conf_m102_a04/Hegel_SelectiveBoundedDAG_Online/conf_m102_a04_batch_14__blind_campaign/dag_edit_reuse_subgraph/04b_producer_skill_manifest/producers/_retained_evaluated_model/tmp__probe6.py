import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
def F(*x):return np.column_stack((np.ones(len(x[0])),*x))
def grid(fun,low=.1,hi=8):
 g=np.linspace(low,hi,121); i=np.argmin([fun(q) for q in g]);return minimize_scalar(fun,bounds=(g[max(0,i-1)],g[min(len(g)-1,i+1)]),method='bounded').x
u,v=map(s['panel_a'].__getitem__,('u','v')); keep=np.linalg.norm(v,axis=1)<5; print('coord keep',sum(keep),'ranges',u.min(),u.max(),v[keep].min(),v[keep].max());u=u[keep];v=v[keep];o=np.roll(u,2,axis=1)
for name,fun in {'sin':np.sin,'tanh':np.tanh,'atan':np.arctan}.items():
 def fit(q,opts,verbose=False):
  vals=[]
  for j in range(4):
   a,b,y=u[:,j],o[:,j],v[:,j]; poss=[a,b,fun(q[0]*b),fun(q[-1]*a),a**3,b**3,a*b, a*a*b, a*b*b]; f=F(*(poss[k] for k in opts));c=ls(f,y);vals.append(f@c-y)
   if verbose: print(np.round(c,5),end=' ')
  return np.stack(vals).ravel()
 for opts in [(0,2),(0,2,4),(0,2,3),(0,1,2,3),(0,2,3,6),(0,2,3,7,8)]:
  r=least_squares(lambda q:fit(q,opts),[2.2,1.6],bounds=(.1,6),loss='soft_l1',f_scale=.003); print(name,opts,np.round(r.x,5),np.round(np.quantile(abs(r.fun),[.5,.8,.9,1]),6));fit(r.x,opts,True); print()
u=s['panel_c']['u']; r=np.roll(u,-1,axis=1); o=np.roll(u,2,axis=1)
sets={'u':u,'cyclic':u*r,'opposite':(u*o)[:,:2],'next+':u+r,'next-':u-r,'opposite+':(u+o)[:,:2],'opposite-':(u-o)[:,:2],'square':u*u}
for key,y in [('v1',s['panel_c']['v1']),('v2',s['panel_c']['v2'])]:
 print(key)
 res=[]
 base=F(u,u**3,np.arctan(1.4*u)) if key=='v1' else F(u)
 for n,w in sets.items():
  for name,fun in {'linear':lambda x:x,'cos':np.cos,'sin':np.sin,'atan':np.arctan,'cosoff':lambda x:np.cos(x+.4),'rat':lambda x:1/(1+x*x)}.items():
   for j,b in enumerate([np.ones((len(u),1)),base]):
    def f(h):return np.c_[b,fun(h*w)]
    def loss(q):
     a=f(q);return np.linalg.norm(y-a@ls(a,y))
    q=grid(loss); a=f(q);h=a@np.linalg.pinv(a);e=(y-h@y)/np.maximum(1-h.diagonal()[:,None],1e-8);res.append((np.mean(abs(e)),loss(q),n,name,j,q))
 print(*sorted(res)[:22],sep='\n')
for key in ['panel_a','panel_b','panel_c','train','test']:
 print(key,{k:(v.shape,np.round(np.min(v),3),np.round(np.max(v),3)) for k,v in s[key].items()})
