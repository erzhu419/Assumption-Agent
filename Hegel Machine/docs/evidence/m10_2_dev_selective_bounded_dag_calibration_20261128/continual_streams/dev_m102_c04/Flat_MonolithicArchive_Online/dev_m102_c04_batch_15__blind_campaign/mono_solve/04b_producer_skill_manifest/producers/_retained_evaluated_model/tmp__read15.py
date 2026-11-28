exec(open('/tmp/explore15.py').read().split('for j in (1,2,3):')[0])
from scipy.spatial.distance import cdist
P=list(it.combinations(range(4),2)); pairs=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in P]); roll=lambda x:np.roll(x,-1,axis=1)
d=lambda x:aff(x,x*roll(x),x**3)
w=ls(d(u),v);print('A coef',np.round(w,6)); print('A residual selection',np.sort(np.max(abs(d(s['panel_a']['u'])@w-s['panel_a']['v']),axis=1))[-8:]);w[abs(w)<.001]=0
x=tr['x'];z=x.copy();j=np.arange(4)
for _ in range(45):
 J=np.broadcast_to(w[1:5].T,(len(z),4,4)).copy();J[:,j,j]+=w[5:9].diagonal()*roll(z)+3*w[9:].diagonal()*z*z;J[:,j,(j+1)%4]+=w[5:9].diagonal()*z; z-=np.clip(np.linalg.solve(J,(d(z)@w-x)[:,:,None])[:,:,0],-.4,.4)
print('inv check',abs(d(z)@w-x).max(),z.min(0),z.max(0));
U,V,W=s['panel_c'].values();sat=lambda z,k:z/(1+k*abs(z));root=lambda z,k:z/np.sqrt(1+k*z*z)
M={}
for key,F,T in [('p',sat,V),('q',root,W)]:
 def d0(x,k):return aff(x,F(x,k),pairs(x))
 o=minimize_scalar(lambda k:rm(d0(U,k)@ls(d0(U,k),T)-T),bounds=(.1,2),method='bounded'); ww=ls(d0(U,o.x),T); print(key,o.x,o.fun,np.round(ww,4));M[key]=d0(z,o.x)@ww
M['z']=z
B,T=s['panel_b'].values(); fs={'sat':sat,'root':root,'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'log':lambda x,k:np.log1p(k*x*x),'rat':lambda x,k:x/(1+k*x*x),'sq':lambda x,k:x*x,'cube':lambda x,k:x**3,'exp':lambda x,k:np.exp(-k*x*x)}
out=[]
for nm,F in fs.items():
 for mode in range(4):
  def bf(x,k):return aff(*[(F(x,k),),(x,F(x,k)),(F(x,k),pairs(x)),(x,F(x,k),pairs(x))][mode]);
  o=minimize_scalar(lambda k:rm(bf(B,k)@ls(bf(B,k),T)-T),bounds=(.03,5),method='bounded');out.append((o.fun,nm,mode,o.x))
print('B best',sorted(out)[:12]);
for bi,(_,nm,mode,k) in enumerate(sorted(out)[:3]):
 F=fs[nm]; bf=lambda x:aff(*[(F(x,k),),(x,F(x,k)),(F(x,k),pairs(x)),(x,F(x,k),pairs(x))][mode]);M['b'+str(bi)]=bf(z)@ls(bf(B),T)
t=s['eval_times']; Y=tr['traj']; bas=lambda k:-np.expm1(-np.outer(t,k)); opt=least_squares(lambda k:(bas(k)@ls(bas(k),Y[:52].T)-Y[:52].T).ravel(),[1.8,.43],bounds=([.8,.03],[8,.79])); Utime=bas(opt.x);print('rates',opt.x,rm(opt.fun));A=ls(Utime,Y.T).T;R=np.c_[tr['y'],A];
def report(f):
 pred=f@ls(f[:52],R[:52]);return np.r_[np.mean(abs(pred[52:]-R[52:]),axis=0),np.mean(abs((pred[52:,1:]@Utime.T)-Y[52:])),np.mean(abs(pred[:52]-R[:52]),axis=0)]
features=[]
for n,a in M.items():
 for m,b in M.items():
  if m<n:continue
  cross=(a[:,:,None]*b[:,None,:]).reshape(len(a),16) if m!=n else np.c_[a*a,pairs(a)];
  for mode,c in [('lin',aff(a,b)),('dot',aff(a,b,np.sum(a*b,axis=1))),('all',aff(a,b,cross)),('sq',aff(a,b,cross,a*a,b*b))]:
   e=report(c);features.append((max(e[0],e[3]),n,m,mode,np.round(e,5)))
  common=aff(a,b,np.sum(a*b,axis=1)); scores=[]
  for i in range(cross.shape[1]):
   c=np.c_[common,cross[:,i]]; rr=R[:52]-c[:52]@ls(c[:52],R[:52]);scores.append(rm(rr/R[:52].std(0)))
  j=np.argmin(scores);e=report(np.c_[common,cross[:,j]]);features.append((max(e[0],e[3]),n,m,'cross'+str(j),np.round(e,5)))
print('readout top');[print(row) for row in sorted(features,key=lambda q:q[0])[:22]]
np.savez('/tmp/dat15.npz',**M,R=R,time=Utime,Y=Y)
