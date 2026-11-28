exec(open('/tmp/probe_panels.py').read().split('for k,panel')[0])
from scipy.optimize import minimize_scalar
B=s['panel_b']; C=s['panel_c']; funcs={
 'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+k*x*x),'sinh':lambda x,k:np.sinh(k*x),'cube':lambda x,k:x**3+k*x,'signed2':lambda x,k:x*abs(x)+k*x,'exp':lambda x,k:np.exp(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'atan':lambda x,k:np.arctan(k*x),'sat':lambda x,k:x/np.sqrt(1+k*x*x),'sin3':lambda x,k:x**3+k*np.sin(x),'sq':lambda x,k:x+k*x*x,'exp2':lambda x,k:np.exp(k*x*x),'cos':lambda x,k:np.cos(k*x),'ssq':lambda x,k:np.sign(x)*abs(x)**k}
for tag,panel,key in [('B',B,'v'),('C1',C,'v1'),('C2',C,'v2')]:
 x=panel['u']; v=panel[key]; best=[]
 for name,f in funcs.items():
  for mode in range(5):
   def design(x,k):
    w=f(x,k); return np.c_[np.ones(len(x)),w,*([x] if mode in [1,3] else []),*([x*x] if mode in [2,3] else []),*([w*w] if mode==4 else [])]
   def loss(k):
    X=design(x,k); H=X@np.linalg.pinv(X,rcond=1e-9); return np.mean(abs((v-H@v)/np.clip(1-H.diagonal(),1e-7,None)[:,None]))
   grid=np.linspace(-.99 if name in ['cube','sq','signed2','sin3'] else .05,5,45); i=np.argmin([loss(k) for k in grid]); opt=minimize_scalar(loss,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); D=design(x,opt.x); w=design(u,opt.x)@reg(D,v); e=opt.fun; a,b,c,d=w.T; best.append((e,name,mode,opt.x,mse(D,v),scores(np.c_[poly(w,1),a*c+b*d],y)[1],scores(poly(w,2),y)[1],scores(poly(w,1),T)[1],scores(poly(w,2),T)[1]));
 print(tag,*sorted(best)[:12],sep='\n');print('label top',*sorted(best,key=lambda a:a[6]+a[7])[:4],sep='\n')
