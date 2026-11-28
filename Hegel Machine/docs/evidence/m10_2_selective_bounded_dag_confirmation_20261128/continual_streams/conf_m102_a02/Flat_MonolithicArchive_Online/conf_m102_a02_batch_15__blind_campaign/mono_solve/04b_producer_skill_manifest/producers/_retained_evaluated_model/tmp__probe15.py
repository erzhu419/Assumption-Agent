import pickle,numpy as n
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=n.load('/tmp/modeldata.npz'); ls=lambda x,y:n.linalg.lstsq(x,y,rcond=None)[0]; A=lambda x:n.c_[n.ones(len(x)),x]; ii,jj=n.triu_indices(4,1)
funs={'cos':lambda x,p:n.cos(p*x),'sin':lambda x,p:n.sin(p*x),'exp':lambda x,p:n.exp(-p*x*x),'rat':lambda x,p:1/(1+p*x*x),'tanh':lambda x,p:n.tanh(p*x),'even_tanh':lambda x,p:n.tanh(p*x*x),'oddrat':lambda x,p:x/(1+p*x*x),'oddexp':lambda x,p:x*n.exp(-p*x*x),'pow':lambda x,p:abs(x)**p,'oddpow':lambda x,p:n.sign(x)*abs(x)**p,'hinge':lambda x,p:n.maximum(abs(x)-p,0),'coscube':lambda x,p:n.cos(p*x**3),'cosabs2':lambda x,p:n.cos(p*x*x)}
for name,x,y in [(k+v,s[k]['u'],s[k][v]) for k in ('panel_b','panel_c') for v in s[k] if v!='u']+[('amp',d['x'],d['amp']),('y',d['x'],d['y'][:,None])]:
 print('\n',name); rows=[]
 for basekind in (0,1):
  base=A(x) if not basekind else n.c_[A(x),x[:,ii]*x[:,jj]]
  for key,f in funs.items():
   def des(p):return n.c_[base,f(x,p)]
   end=1.1 if key=='hinge' else 12
   def loss(p):
    a=des(p); return n.mean((a@ls(a,y)-y)**2)
   g=n.linspace(.015,end,121); i=n.argmin([loss(p) for p in g]); p=minimize_scalar(loss,bounds=(g[max(0,i-1)],g[min(len(g)-1,i+1)]),method='bounded').x; a=des(p); mse=loss(p); loo=n.mean(abs((a@ls(a,y)-y)/(1-(a*n.linalg.pinv(a).T).sum(1))[:,None])); rows.append((loo,key,basekind,p,mse**.5))
 print(sorted(rows)[:12]);
 for zz in (A(x),n.c_[A(x),x*x],n.c_[A(x),x*x,x[:,ii]*x[:,jj]]):
  w=ls(zz,y); print('poly',zz.shape[1],n.mean(abs(zz@w-y),axis=0),n.round(w.T,2))
