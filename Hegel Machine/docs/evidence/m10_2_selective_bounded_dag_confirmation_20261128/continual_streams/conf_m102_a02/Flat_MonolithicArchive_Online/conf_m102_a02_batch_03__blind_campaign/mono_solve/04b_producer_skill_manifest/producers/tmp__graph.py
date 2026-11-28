from probe import *
from scipy.optimize import least_squares

def nextcol(x): return np.roll(x,-1,axis=1)
def cal():
 p=s['panel_a']; ok=np.linalg.norm(p['v'],axis=1)<8; u,v=p['u'][ok],p['v'][ok]; n=nextcol(u)
 w=np.array([solve(np.c_[np.ones(len(u)),a,b,a*b],c) for a,b,c in zip(u.T,n.T,v.T)])
 def apply(x): return w[:,0]+w[:,1]*x+w[:,2]*nextcol(x)+w[:,3]*x*nextcol(x)
 def inv(x):
  d=(x-w[:,0])/w[:,1]
  return least_squares(lambda a:(apply(a.reshape(x.shape))-x).ravel(),d.ravel(),jac_sparsity=np.kron(np.eye(len(x),dtype=bool),np.eye(4,dtype=bool)|nextcol(np.eye(4,dtype=bool))).T, max_nfev=100).x.reshape(x.shape)
 # Explicit 4x4 batched Newton Jacobians.
 def inv(x):
  d=(x-w[:,0])/w[:,1]; ii=np.arange(4)
  for _ in range(18):
   J=np.zeros((len(x),4,4)); J[:,ii,ii]=w[:,1]+w[:,3]*nextcol(d); J[:,ii,(ii+1)%4]=w[:,2]+w[:,3]*d
   dd=np.linalg.solve(J,(apply(d)-x)[...,None])[...,0]; d-=np.clip(dd,-.5,.5)
  return d
 print('A',w,'rmse',np.std(apply(u)-v,0),'worst',np.max(abs(apply(u)-v),0));
 for k in range(4):
  a,b=u[:,k],n[:,k]; r=v[:,k]-apply(u)[:,k]; cand={'a2':a*a,'b2':b*b,'a3':a**3,'b3':b**3,'ab2':a*b*b,'a2b':a*a*b,'ab3':a*b**3,'three':a*b*n[:,(k+1)%4]}; F=np.c_[np.ones(len(u)),a,b,a*b]; print('A residual',k,[(key,round(np.corrcoef(r,g-F@solve(F,g))[0,1],3)) for key,g in cand.items()]);
 return apply,inv

def operators():
 funcs={}
 for key in ['panel_b','panel_c']:
  p=s[key]; x=p['u'];
  for lab,y in p.items():
   if lab=='u':continue
   print(key,lab,'aff, quad rms',np.std(aff(x)@solve(aff(x),y)-y,0),np.std(pp(x)@solve(pp(x),y)-y,0))
   print('diag quad',[np.round(solve(np.c_[aff(x),x*x],y),3)])
   name=key[-1]+lab; funcs[name+'q']=lambda z,w=solve(np.c_[aff(x),x*x],y):np.c_[aff(z),z*z]@w
   funcs[name+'Q']=lambda z,w=solve(pp(x),y):pp(z)@w
   for typ in ['tanh','sin','root','rat','gauss']:
    fun={'tanh':np.tanh,'sin':np.sin,'root':lambda z:z/np.sqrt(1+z*z),'rat':lambda z:z/(1+z*z),'gauss':lambda z:z*np.exp(-z*z)}[typ]; results=[]; W=[]; C=[]
    for j in range(4):
     b=x[:,(j+1)%4]; A=aff(x)
     def get(p,z=x,j=j,fun=fun):return np.c_[aff(z),fun(p[0]*z[:,(j+1)%4]+p[1])]
     def res(p): F=get(p);return F@solve(F,y[:,j])-y[:,j]
     p=least_squares(res,[1,0],bounds=([.05,-2],[4,2])).x; W.append(solve(get(p),y[:,j])); C.append(p); results.append(np.std(res(p)))
    print(name,typ,np.round(results,4),'params',np.round(C,3))
    funcs[name+typ]=lambda z,C=C,W=W,get=get:np.column_stack([get(C[j],z,j)@W[j] for j in range(4)])
 return funcs

if __name__=='__main__':
 ap,iv=cal(); ops=operators(); x=s['train']['x']; T=np.c_[s['train']['y'],s['train']['traj']]; rows=[]
 def evaluate(F,lab,r=0.):
  w=solve(F[:52],T[:52]) if r==0 else np.linalg.solve(F[:52].T@F[:52]+r*np.eye(F.shape[1]),F[:52].T@T[:52]); e=F@w-T; rows.append((lab,np.mean(abs(e[52:,0])),np.mean(abs(e[52:,1:])),np.std(e[:52,0]),np.std(e[:52,1:])))
 for name,z in [('raw',x),('forward',ap(x)),('inv',iv(x))]:
  print(name,'bounds',z.min(0),z.max(0),'inverse check',np.max(abs(ap(iv(x))-x)))
  evaluate(aff(z),name+' aff');evaluate(pp(z),name+' quad');
  for lab,op in ops.items():
   g=op(z);evaluate(aff(g),name+' '+lab);evaluate(np.c_[aff(z),g],name+' z+'+lab);evaluate(pp(g),name+' Q'+lab,.002);evaluate(np.c_[pp(z),g],name+' P+'+lab,.003)
  for typ,u in [('rat',z/(1+z*z)),('tanh',np.tanh(z)),('gauss',z*np.exp(-z*z))]:
   evaluate(np.c_[aff(z),u],name+' '+typ);evaluate(np.c_[pp(z),u],name+' P'+typ,.01); evaluate(np.c_[pp(z),u,(z[:,:,None]*u[:,None,:]).reshape(len(z),-1)],name+' PU'+typ,.03)
 print('best y',*sorted(rows,key=lambda r:r[1])[:15],sep='\n'); print('best traj',*sorted(rows,key=lambda r:r[2])[:15],sep='\n')
