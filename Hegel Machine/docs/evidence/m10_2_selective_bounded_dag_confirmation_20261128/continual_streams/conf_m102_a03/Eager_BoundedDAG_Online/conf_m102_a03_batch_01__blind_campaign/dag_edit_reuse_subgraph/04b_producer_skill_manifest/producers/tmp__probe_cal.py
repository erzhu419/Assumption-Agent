import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=120)
def poly(x,d):
 t=[t for k in range(d+1) for t in itertools.combinations_with_replacement(range(x.shape[1]),k)]
 return np.column_stack([np.prod(x[:,q],axis=1) for q in t]),t
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def stats(a,y):
 p=np.linalg.pinv(a); c=p@y; e=y-a@c; loo=e/(1-(a*p.T).sum(1))[:,None]; return c,np.sqrt(np.mean(e**2)),np.sqrt(np.mean(loo**2)),np.max(np.abs(loo))
for key in ('panel_a','panel_b','panel_c'):
 p=s[key]; x=p['u'];
 for col in set(p)-{'u'}:
  y=p[col]; good=np.max(abs(y),axis=1)<3; x1,y1=x[good],y[good]; print(key,col,len(x1))
  for name,u,v in [('forward',x1,y1),('inverse',y1,x1)]:
   for d in range(1,4):
    a,t=poly(u,d)
    if a.shape[1]>=len(u): continue
    c,*r=stats(a,v); print(name,d,r)
    if key=='panel_a' and name=='forward':
     for j in range(4): print(j,[(q,round(c[i,j],5)) for i,q in enumerate(t) if abs(c[i,j])>=.012])
  if key=='panel_a':
   a,t=poly(x1,2); c=fit(a,y1); print('quadratic residual correlations with trig, rational and high degrees')
   e=y1-a@c
   for j in range(4):
    bank=[]
    for i in range(4):
     for k in range(4):
      for sg in [-1,0,1]:
       z=x1[:,i]+sg*x1[:,k]
       for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('cube',lambda t:t**3),('rat',lambda t:t/(1+t*t)),('abs',np.abs),('rat2',lambda t:1/(1+t*t))]:
        for w in ([.5,1.,1.25,2.] if fn in ('sin','cos','rat','rat2','tanh') else [1.]):
         b=f(w*z)
         for mult in [-1,j]:
          v=b if mult==-1 else b*x1[:,mult]; v=v-a@fit(a,v); norm=np.linalg.norm(v)*np.linalg.norm(e[:,j]);
          if norm>1e-8:bank.append((abs(v@e[:,j])/norm,(i,k,sg,fn,w,mult)))
    print(j,sorted(bank,reverse=True)[:12])
