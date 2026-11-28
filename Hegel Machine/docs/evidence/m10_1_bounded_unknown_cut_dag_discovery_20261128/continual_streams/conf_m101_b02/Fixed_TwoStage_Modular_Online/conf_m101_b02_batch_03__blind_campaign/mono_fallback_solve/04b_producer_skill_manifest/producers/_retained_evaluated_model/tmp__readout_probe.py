import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations
from sklearn.linear_model import Lasso
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; t=s['eval_times']; T=s['train']['traj'];y=s['train']['y']; N=52
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
D=lambda *a:np.column_stack([np.ones(len(a[0])),*a])
U=-np.expm1(-t[:,None]*np.array([.41837,1.81795])); H=ls(U,T.T).T; Y=np.c_[y,H]; scale=Y[:N].std(0); Y=Y/scale
p=s['panel_c'];u=p['u'];
funcs={'oddexp':lambda x,k:-np.sign(x)*np.expm1(-k*abs(x)),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'even':lambda x,k:x*x/np.sqrt(1+k*x*x),'cos':lambda x,k:np.cos(k*x),'rat':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'gauss':lambda x,k:np.exp(-k*x*x)}
ops=[]
for name in ['v1','v2']:
 v=p[name]; cand=[]
 for n,f in funcs.items():
  for lin in [False,True]:
   df=lambda x,k:D(x,f(x,k)) if lin else D(f(x,k)); loss=lambda k:np.linalg.norm(df(u,k)@ls(df(u,k),v)-v); o=minimize_scalar(loss,bounds=(.02,4),method='bounded'); cand.append((o.fun,n,lin,o.x))
 cand.sort(); print(name,cand[:6]);_,n,lin,k=cand[0];f=funcs[n]; df=lambda x,k,f=f,lin=lin:D(x,f(x,k)) if lin else D(f(x,k));w=ls(df(u,k),v);ops.append(lambda x,df=df,k=k,w=w:df(x,k)@w)
b=s['panel_b'];B=ls(D(b['u']),b['v']); blocks={'x':x,'b':D(x)@B,'bi':(x-B[0])@np.linalg.inv(B[1:])};
pa=s['panel_a']; a=pa['u']; v=pa['v']; A=lambda a:D(a,a*a,a**3,np.array([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]).T);F=A(a); W=ls(F,v)
for _ in range(70):
 q=np.maximum(np.linalg.norm(v-F@W,axis=1),.00025)**-.5;W=ls(F*q[:,None],v*q[:,None]);
blocks['a']=A(x)@W
for name,z in list(blocks.items()):
 c,d=[op(z) for op in ops]; blocks['c'+name]=c;blocks['d'+name]=d
 def check(tag,g):
  e=Y-g@ls(g[:N],Y[:N]);print(tag, np.round(np.mean(abs(e[:N]),0)*scale,4),np.round(np.mean(abs(e[N:]),0)*scale,4))
 check(name+' c d',D(z,c,d)); check(name+' c d diag',D(z,c,d,z*c,z*d,c*d));
 for key,v in [('z',z),('c',c),('d',d)]:
  print(name+key,'corr',np.round(np.corrcoef(np.c_[v,Y].T)[:4,-3:],2).T)
features={}
for n,z in blocks.items():
 for i,v in enumerate(z.T):
  for fname,h in [('id',v),('sq',v*v),('cub',v**3)]+[(key+str(k),f(v,k)) for key,f in funcs.items() for k in [.35,.7,1.2,2.]]: features[f'{n}{i}:{fname}']=h
for n in ['x','b','bi','a']:
 for l,r in [(n,n),('c'+n,'c'+n),('d'+n,'d'+n),(n,'c'+n),(n,'d'+n),('c'+n,'d'+n)]:
  for i in range(4):
   for j in range(4):
    if l!=r or i<j:
     a,b=blocks[l][:,i],blocks[r][:,j];features[f'{l}{i}*{r}{j}']=a*b
     if l==r:
      for sign in [-1,1]:
       h=a+sign*b
       for nm,f in [('s',np.sin),('t',np.tanh),('r',lambda h:h/(1+h*h))]:features[f'{nm}({l}{i}{sign:+}{r}{j})']=f(h)
F=np.column_stack(list(features.values())); Fs=(F-F[:N].mean(0))/np.maximum(F[:N].std(0),1e-9); keys=list(features)
for target in range(3):
 for alpha in [.08,.025,.008,.002]:
  model=Lasso(alpha=alpha,max_iter=6000,tol=.0001).fit(Fs[:N],Y[:N,target]); idx=np.argsort(abs(model.coef_))[-9:][::-1]; e=abs(Y[:,target]-model.predict(Fs))*scale[target]; print('lasso',target,alpha,np.round([e[:N].mean(),e[N:].mean()],4),[(keys[i],round(model.coef_[i],2)) for i in idx if model.coef_[i]])
