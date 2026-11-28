exec(open('/tmp/graph101.py').read().split('for key,p,v in')[0])
from scipy.special import erf
u=s['panel_b']['u']; v=s['panel_b']['v']; w=(x-off)/sc
families={'exp':lambda q,k:q*(-np.expm1(-k*q*q)), 'rat':lambda q,k:q**3/(1+k*q*q), 'tanh':lambda q,k:np.tanh(k*q), 'asinh':lambda q,k:np.arcsinh(k*q), 'erf':lambda q,k:erf(k*q), 'sin':lambda q,k:np.sin(k*q), 'cube':lambda q,k:q**3, 'poly':lambda q,k:q**3+k*q**5, 'ratp':lambda q,k:q**3/(1+k[0]*abs(q)**k[1])}
P=lambda z:np.c_[O(z),z[:,0]*z[:,2],z[:,1]*z[:,3]]
B=lambda z:O(z)
for name,f in families.items():
 for mode in ['base','shift','even','offset','post']:
  d=2 if name=='ratp' else 1
  def H(q,p):
   qq=q+(p[d+1:] if mode=='shift' else 0); k=p[:d] if d>1 else p[0]; h=f(qq,k)+p[d]*qq
   if mode=='even':h=h+p[d+1]*qq**2
   if mode=='offset':h=h+p[d+1]*qq**4
   return O(h)
  n=d+1+{'base':0,'shift':4,'even':1,'offset':1,'post':1}[mode]
  def fit(i,weight):
   def eval(p,predict=False):
    a=H(u,p); target=v if mode!='post' else v+p[-1]*v**3; c=L(a,target); z=H(w,p)@c
    if mode=='post':
     for _ in range(4): z-=(z+p[-1]*z**3-H(w,p)@c)/(1+3*p[-1]*z*z)
    A=P(z); D=B(z); y=A@L(A[i],Y[i,0]); ac=L(D[i],Y[i,1:]); U,S,V=np.linalg.svd(ac,full_matrices=False); ac=(U[:,:2]*S[:2])@V[:2]; pred=np.c_[y,D@ac]; r=(a@c-target).ravel()
    return (pred,np.sqrt(np.mean(r*r))) if predict else np.r_[r,weight*(pred[i]-Y[i]).ravel()]
   lo=np.r_[[.015],([.6] if d==2 else []),[-4],np.full(n-d-1,0 if mode=='post' else -.5)]; hi=np.r_[[5],([5.] if d==2 else []),[4],np.full(n-d-1,.5)]; starts=[np.r_[[k],([2.] if d==2 else []),[linear],np.full(n-d-1,.001 if mode=='post' else 0)] for k,linear in [(.6,-.08),(1.3,-.08),(2.5,0),(.8,.2)]]
   o=min([least_squares(eval,p,bounds=(lo,hi),max_nfev=70) for p in starts],key=lambda f:f.cost);return (*eval(o.x,True),o.x)
  for weight in [0,.7,3]:
   pp=[]; mm=[]
   for i,j in folds:
    pred,rm,p=fit(i,weight); e=abs(pred[j]-Y[j]); pp.append((e[:,0].mean(),e[:,1:].mean())); mm.append(rm)
   print(name,mode,weight,np.round(pp[0],5),np.round(np.max(pp,0),5),round(np.mean(mm),6),p,flush=True)
