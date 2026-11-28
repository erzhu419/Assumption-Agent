exec(open('/tmp/probe.py').read().split('for key in')[0]); from scipy.optimize import root
np.set_printoptions(precision=5,suppress=True)
def fitop(u,v,f):
 d=f(u); b=ls(d,v); print('panel fit',np.std(d@b-v)); return lambda x:f(x)@b
p=s['panel_a']; u,v=p['u'],p['v']; basis=lambda x:np.stack((np.ones_like(x),x,np.roll(x,-1,axis=-1),x*np.roll(x,-1,axis=-1),x**3),axis=-1)
d=basis(u); k=np.ones(len(u),bool)
for i in range(10):
 a=np.array([ls(d[k,j],v[k,j]) for j in range(4)]); e=np.linalg.norm((d*a).sum(-1)-v,axis=1); k=e<max(.001,8*np.median(e)); print(i,k.sum(),np.median(e),np.max(e[k]))
print('A',a); x=np.r_[tr['x'],s['test']['x']]; z=np.array([root(lambda q:(basis(q)*a).sum(-1)-r,r).x for r in x]);print('inverse error',np.max(abs((basis(z)*a).sum(-1)-x)))
p=s['panel_b']; B=fitop(p['u'],p['v'],lambda q:aff(dead(q,[.35,1.5])));p=s['panel_c']; C=fitop(p['u'],p['v1'],lambda q:aff(np.column_stack([q[:,i]*q[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])))
u,v=p['u'],p['v2']; funcs={'polynomial':(lambda q,b:q+b[0]*q*q+b[1]*q**3,[.4,.05],[-3,-3],[3,3]),'rational':(lambda q,b:q/(1+b[0]*q+b[1]*q*q),[-.4,.2],[-1,0],[1,2]),'exp':(lambda q,b:np.expm1(b[0]*q),[.8],[.01],[4]),'softplus':(lambda q,b:np.logaddexp(0,b[0]*q+b[1]),[1,0],[.01,-5],[10,5]),'tanh':(lambda q,b:np.tanh(b[0]*q+b[1]),[.3,-.7],[.01,-5],[5,5]),'sigmoid':(lambda q,b:1/(1+np.exp(-b[0]*q-b[1])),[1,-1],[.01,-10],[8,10]),'sin':(lambda q,b:np.sin(b[0]*q+b[1]),[.3,-.7],[.01,-3],[5,3])}
ops={};
for name,(f,b,l,h) in funcs.items():
 def res(b):
  d=aff(f(u,b));return (d@ls(d,v)-v).ravel()
 opt=least_squares(res,b,bounds=(l,h),max_nfev=300); print('C2',name,opt.x,np.std(opt.fun));ops[name]=fitop(u,v,lambda q,f=f,b=opt.x:aff(f(q,b)))
def score(d,t):
 b=ls(d[:n],t[:n]);return np.mean(abs(d[:n]@b-t[:n])),np.mean(abs(d[n:72]@b-t[n:]))
Fs={'id':aff,'sq':lambda z:aff(np.c_[z,pairs(z)]),'pair':lambda z:aff(np.c_[z,z[:,0]*z[:,2],z[:,1]*z[:,3]]),'cub':lambda z:aff(np.c_[z,pairs(z),pairs(z,3)])}
lat={'x':x,'A':z,'B':B(x),'AB':B(z),'AC':C(z),'ABC':C(B(z)),'BC':C(B(x))};
for name,D in ops.items():
 for k,q in [('A',z),('AB',B(z)),('ABC',C(B(z))),('AC',C(z))]:lat[k+'D'+name]=D(q)
for out in ['y','traj']:
 r=sorted((score(f(q),tr[out])[1],score(f(q),tr[out])[0],k,j) for k,q in lat.items() for j,f in Fs.items()); print(out,r[:17]); print('simple',[(k,[tuple(round(f,5) for f in score(Fs[j](lat[k]),tr[out])) for j in ['id','pair','sq']]) for k in list(lat)[:7]])
