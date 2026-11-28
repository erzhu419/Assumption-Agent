import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; cyc=u*np.roll(u,-1,axis=1)
ls=lambda f,y:np.linalg.lstsq(f,y,rcond=1e-10)[0]
one=lambda *f:np.column_stack((np.ones(len(u)),*f))
def score(f,y):
 r=f@ls(f,y)-y
 h=np.sum(f*np.linalg.pinv(f).T,axis=1)
 return np.sqrt(np.mean(r*r)), np.sqrt(np.mean((r/np.maximum(1-h[:,None],1e-9))**2))
for y,baseopts,cands in [(v1,{'u':(u,), 'u3':(u,u**3),'uabs':(u,u*abs(u)),'empty':()}, {'soft':lambda a,q:a/(1+q*abs(a)),'arctan':lambda a,q:np.arctan(q*a),'rational':lambda a,q:a/(1+q*a*a),'tanh':lambda a,q:np.tanh(q*a),'asinh':lambda a,q:np.arcsinh(q*a)}),(v2,{'u':(u,),'u2':(u,u*u),'cyc':(u,cyc),'u2cyc':(u,u*u,cyc),'empty':()}, {'rational2':lambda a,q:a*a/(1+q*a*a),'cos':lambda a,q:np.cos(q*a),'rational':lambda a,q:a/(1+q*a*a),'cosm':lambda a,q:np.cos(q*a),'soft':lambda a,q:a/(1+q*abs(a))})]:
 out=[]
 for b,f in baseopts.items():
  for a,x in {'u':u,'cyc':cyc,'diff':u-np.roll(u,-1,axis=1),'sum':u+np.roll(u,-1,axis=1)}.items():
   for nm,fn in cands.items():
    if nm=='cosm':continue
    g=lambda q:one(*f,fn(x,q))
    op=minimize_scalar(lambda q:score(g(q),y)[1],bounds=(.02,5),method='bounded')
    out.append((op.fun,*score(g(op.x),y)[:1],b,a,nm,op.x))
 print('TOP',*[tuple(round(i,8) if isinstance(i,float) else i for i in z) for z in sorted(out)[:14]],sep='\n')
for name,f in [('u cyc',one(u,cyc)),('cyc',one(cyc)),('u u2 cyc',one(u,u*u,cyc))]:
 print(name,score(f,v2),ls(f,v2).round(3))
