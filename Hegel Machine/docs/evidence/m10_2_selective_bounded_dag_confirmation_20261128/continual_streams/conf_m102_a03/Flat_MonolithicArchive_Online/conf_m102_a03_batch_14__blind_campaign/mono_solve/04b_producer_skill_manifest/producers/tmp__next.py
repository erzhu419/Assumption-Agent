exec(open('/tmp/graph_probe.py').read().split('B=panel(')[0]); np.set_printoptions(precision=5,suppress=True)
u=s['panel_b']['u']; v=s['panel_b']['v']; S=np.load('/tmp/spaces.npz'); models=[]
for name,fun,lo,hi,starts in [
 ('warp',lambda x,p:x*(1+p[0]*abs(x)**p[1])**p[2],[.0001,.3,-5],[100,6,3],[[1,2,-1],[3,1,-.5],[10,3,-1],[1,2,.5]]),
 ('satplus',lambda x,p:x+p[0]*np.tanh(p[1]*x)+p[2]*np.arcsinh(p[3]*x),[-8,.1,-8,.1],[8,20,8,50],[[1,5,-1,5],[-1,5,1,5],[1,2,-1,10]]),
 ('threshold',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[0,.1],[1,6],[[.35,1.5],[.1,.5],[.5,2]]),
 ('poly5',lambda x,p:x+p[0]*x**3+p[1]*x**5,[-10,-10],[10,10],[[0,0]]),
 ('powerblend',lambda x,p:x+p[0]*np.sign(x)*abs(x)**p[1]+p[2]*np.sign(x)*abs(x)**p[3],[-30,.1,-30,.1],[30,8,30,8],[[-1,.8,1,3],[1,.4,1,2],[-1,2,1,4]])]:
 for aug in [False,True]:
  if aug: f=lambda x,p: af(np.c_[x,fun(x,p)])
  else: f=lambda x,p: af(fun(x,p))
  def res(p):
   D=f(u,p); return (D@fit(D,v)-v).ravel()
  q=min([least_squares(res,p,bounds=(lo,hi),max_nfev=250) for p in starts],key=lambda q:q.cost); a=f(u,q.x); c=fit(a,v); b=f(z,q.x)@c; models.append((np.std(q.fun),name+str(aug),b)); print(name,aug,q.x,np.std(q.fun),flush=True)
models.sort(key=lambda x:x[0]); np.savez('/tmp/newspaces.npz',**{k:b for _,k,b in models});
from sklearn.linear_model import Ridge,OrthogonalMatchingPursuit,Lasso
best=[]; n=52; vfn=lambda x:np.column_stack(x)
def score(a):return np.mean(abs(a[:52]-y[:52])),np.mean(abs(a[52:72]-y[52:]));
for name,b in [('B0',S['B0']),('B3',S['B3'])]+[(k,b) for _,k,b in models[:4]]:
 pair=z*np.roll(b,1,axis=1); P=pair.sum(1); Z=np.c_[z,b]; L=np.c_[af(b),P]; L2=np.c_[L,z]; U=np.c_[b,z,P]; idx=np.arange(4); e=y-L[:72]@fit(L[:52],y[:52]); tests=[]
 base={'b':b,'z':z,'P':P[:,None],'pair':pair,'b+':b+np.roll(b,1,axis=1),'b-':b-np.roll(b,1,axis=1),'bo+':b+np.roll(b,2,axis=1),'bo-':b-np.roll(b,2,axis=1),'z+':z+np.roll(z,1,axis=1),'z-':z-np.roll(z,1,axis=1),'bhat':(L@fit(L[:52],y[:52]))[:,None]}
 funcs={'sq':lambda x:x*x,'cub':lambda x:x**3,'ab':np.abs,'sinslow':lambda x:np.sin(1.3*x),'sinfast':lambda x:np.sin(2.6*x),'tanh':lambda x:np.tanh(2*x),'soft':lambda x:x/(1+abs(x)),'rat':lambda x:x/(1+x*x),'cos':lambda x:np.cos(1.4*x)}
 for k,q in base.items():
  for f,fn in funcs.items():tests.append((k+' '+f,fn(q)))
 for i in range(U.shape[1]):
  for j in range(i,U.shape[1]):
   tests.append(('prod'+str((i,j)),(U[:,i]*U[:,j])[:,None]))
   for d in range(j,U.shape[1]):tests.append(('cub'+str((i,j,d)),(U[:,i]*U[:,j]*U[:,d])[:,None]))
 for l,A in [('L',L),('L2',L2)]:
  R=[]
  for k,h in tests:
   D=np.c_[A,h]; pred=D@fit(D[:52],y[:52]); R.append((*score(pred),k))
  print('single',name,l,sorted(R)[:9]);
  h=b@fit(L[:52],y[:52])[1:5];
  for style in [0,1,2,3,4]:
   for f,fn in funcs.items():
    def rough(p):
     if style==0: return np.c_[A,fn(b@p[:-1]+p[-1])]
     if style==1: return np.c_[A,fn(np.c_[b,P]@p[:-1]+p[-1])]
     if style==2: return np.c_[A,fn(np.c_[z,b]@p[:-1]+p[-1])]
     if style==3: return np.c_[A,fn(z@p[:-1]+p[-1])]
     return np.c_[A,fn(np.c_[z,b,P]@p[:-1]+p[-1])]
    dim=[4,5,8,4,9][style]; rg=np.random.default_rng(4); starts=[rg.normal(0,.7,dim+1) for _ in range(4)];
    if style<2:starts.append(np.r_[fit(L[:52],y[:52])[1:1+dim],0.]);
    def rr(p):
     D=rough(p)[:52]; return D@fit(D,y[:52])-y[:52]
    q=min((least_squares(rr,p,max_nfev=100,bounds=(-8.,8.)) for p in starts),key=lambda q:q.cost); D=rough(q.x); best.append((*score(D@fit(D[:52],y[:52])),name,l,style,f,q.x));
print('links training',sorted(best,key=lambda x:x[0])[:12]); print('links val',sorted(best,key=lambda x:x[1])[:12]);
