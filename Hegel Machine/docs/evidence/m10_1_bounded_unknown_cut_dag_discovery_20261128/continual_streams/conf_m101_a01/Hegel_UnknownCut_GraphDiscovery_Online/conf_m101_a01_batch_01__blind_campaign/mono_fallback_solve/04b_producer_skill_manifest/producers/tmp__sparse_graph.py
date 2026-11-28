from graph_probe import *
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

def cfit(p,k):
 u=p['u']; v=p[k]; fun=(lambda q,a:np.tanh(a*q)) if k=='v1' else (lambda q,a:1/(1+(a*q)**2))
 def d(q,a):return np.c_[np.ones(len(q)),q,fun(q,a)]
 def err(a):
  h=d(u,a); return (h@np.linalg.lstsq(h,v,rcond=None)[0]-v).ravel()
 a=least_squares(err,[2.2 if k=='v1' else .4],bounds=(.05,5)).x; c=np.linalg.lstsq(d(u,a),v,rcond=None)[0]
 return d,c,a

def kernels(st,x):
 A=affine(st['panel_a']); B=affine(st['panel_b']); q=(x-A[0])/np.diag(A[1:]); b=forward(q,B); c=[]
 for k in ('v1','v2'):
  d,w,a=cfit(st['panel_c'],k); c.append(d(q,a)@w)
 out={}
 for nm,h in [('x',x),('a',q),('b',b),('c',c[0]),('d',c[1])]:
  if nm=='x':
   ds=cdist(h,h,'sqeuclidean')/4
   for l in [.75,.9,1.05,1.2,1.4,1.7]:out['rbf'+str(l)]=np.exp(-ds/(2*l*l))
  z=h/np.maximum(.3,h.std(0)); G=z@z.T/4
  for d in (2,3,4,5,6,8):out[nm+'poly'+str(d)]=(1+.5*G)**d
  for tag,g in [('tanh',np.tanh(1.3*h)),('sin',np.sin(1.5*h)),('rat',h/(1+.6*h*h)),('exp',np.exp(.75*h)),('sq',h*h)]:
   g=(g-g.mean(0))/np.maximum(.15,g.std(0)); S=g@g.T/4
   out[nm+tag]=(1+.5*G+.5*S)**3
  if nm not in ('x','a','b'):continue
  delta=z[:,None,:]-z[None,:,:]
  for l in [.65,1.,1.5]:
   fs=np.exp(-delta**2/(2*l*l)); f1=fs.mean(2);f2=sum(fs[:,:,i]*fs[:,:,j] for i,j in combinations(range(4),2))/6;f3=sum(np.prod(fs[:,:,ix],axis=2) for ix in combinations(range(4),3))/4
   out[nm+'anova'+str(l)]=.08*f1+.42*f2+.4*f3+.10*np.prod(fs,axis=2)
   out[nm+'anova2'+str(l)]=.05*f1+.6*f2+.35*f3
 return out

def pred(K,D,y,a,b,n,loo=False):
 Ka=K[np.ix_(a,a)]+n*np.eye(len(a)); L=cho_factor(Ka); w=cho_solve(L,np.c_[D[a],y[a]]); W=w[:,:D.shape[1]]; g=w[:,-1]; F=cho_factor(D[a].T@W); beta=cho_solve(F,D[a].T@g); alpha=g-W@beta; p=D[b]@beta+K[np.ix_(b,a)]@alpha
 if not loo:return p
 diag=np.diag(cho_solve(L,np.eye(len(a))))-np.sum(W*cho_solve(F,W.T).T,axis=1); return p,np.mean(abs(alpha/diag)),alpha/diag

if __name__=='__main__':
 D=poly(x,1); K=kernels(s,x); rng=np.random.default_rng(1124); ix=np.arange(len(y)); more=[]
 for i in range(8):
  p=rng.permutation(ix);more.append((p[:56],p[56:]))
 ss=splits+more; out=[]; pp={}
 for nm,k in K.items():
  for n in [1e-7,1e-5,1e-4,.001,.01]:
   for j in [0,1]:
    p=[pred(k,D,Y[:,j],a,b,n,True) for a,b in ss]; sc=np.array([np.mean(abs(v[0]-Y[b,j])) for v,(a,b) in zip(p,ss)]); ls=np.array([v[1] for v in p]);out.append((j,nm,n,sc,ls));pp[j,nm,n]=p
 for j in [0,1]:
  print('target',j)
  for _,nm,n,v,l in sorted([r for r in out if r[0]==j],key=lambda r:.45*np.mean(r[3])+.25*r[3][0]+.3*np.max(r[3]))[:15]:print(nm,n,np.round(v[:6],5), 'all avg/worst/loo',np.round([v.mean(),v.max(),l.mean()],5))
  options=[r for r in out if r[0]==j];
  for count in [1,5,15,40]:
   e=[]
   for i,(a,b) in enumerate(ss):
    selected=sorted(options,key=lambda r:r[-1][i])[:count]; p=np.mean([pp[j,nm,n][i][0] for _,nm,n,_,_ in selected],axis=0);e.append(np.mean(abs(p-Y[b,j])));
   print('fold LOO ensemble',count,np.round(e,5),'avg/worst',np.mean(e),np.max(e))
