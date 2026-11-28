from probe21 import *
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
A=s['panel_a'];B=s['panel_b'];C=s['panel_c'];tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; n=len(Y)
def rms(a):return np.sqrt(np.mean(a*a))
def fit_a():
 u,v=A['u'],A['v']; good=np.linalg.norm(v,axis=1)<4; print('A excluded',np.where(~good)[0]);
 def fw(a,j,x):return D(x[:,j],np.sin(a*x[:,j]),np.sin(a*x[:,(j+2)%4]))
 def err(a):return np.column_stack([f@reg(f[good],v[good,j])-v[:,j] for j in range(4) for f in [fw(a,j,u)]])[good].ravel()
 o=minimize_scalar(lambda a:rms(err(a)),bounds=(1.5,3.3),method='bounded'); print('A fit',o.x,o.fun,rms(err(2.4))); a=o.x; w=np.column_stack([reg(fw(a,j,u)[good],v[good,j]) for j in range(4)]); print(w.round(4)); return a,w

def invert(x,a,w):
 z=(x-w[0])/w[1];
 for _ in range(25):
  pred=(z*w[1]+np.sin(a*z)*w[2]+np.sin(a*np.roll(z,2,axis=1))*w[3]+w[0]); J=np.zeros((len(z),4,4)); j=np.arange(4); J[:,j,j]=w[1]+a*np.cos(a*z)*w[2]; J[:,j,(j+2)%4]=a*np.cos(a*np.roll(z,2,axis=1))*w[3]; z-=np.clip(np.linalg.solve(J,(pred-x)[...,None])[...,0],-.35,.35)
 return z

def fit_b():
 u,v=B['u'],B['v']; score=[]
 funcs=[('cos',lambda x,a:np.cos(a*x)),('sincos',lambda x,a:np.c_[np.cos(a*x),np.sin(a*x)]),('coscyc',lambda x,a:np.c_[np.cos(a*x),cyc(x)]),('cosu',lambda x,a:np.c_[np.cos(a*x),x]),('phase',lambda x,a:np.c_[np.cos(a*x),np.sin(a*x),cyc(np.cos(a*x)),cyc(np.sin(a*x))])];
 for name,f in funcs:
  def e(a):
   h=D(f(u,a));return rms(h@reg(h,v)-v)
  o=minimize_scalar(e,bounds=(1.8,3.4),method='bounded'); score.append((o.fun,name,f,o.x));print('B',name,o.x,o.fun, e(2.55));
 _,name,f,a=min(score,key=lambda p:p[0]); return f,a,reg(D(f(u,a)),v)

def fit_c():
 u,v=C['u'],C['v2']; funcs=[('trig',lambda x,a:np.c_[np.sin(a*x),np.cos(a*x)]),('quad',lambda x,a:np.c_[x,x*x,a*x**3]),('rational',lambda x,a:np.c_[x/(1+a*x*x),x*x/(1+a*x*x)]),('tanh',lambda x,a:np.c_[np.tanh(a*x),np.tanh(a*x)**2]),('squash',lambda x,a:np.c_[x/(1+a*x*x)**.5,x*x/(1+a*x*x)**.5]),('exp',lambda x,a:np.c_[x,np.exp(a*x)]),('ratmix',lambda x,a:np.c_[x/(1+a*x*x),x*x])];L=[]
 for name,f in funcs:
  def e(a):
   h=D(f(u,a)); q=np.linalg.qr(h)[0];return rms((h@reg(h,v)-v)/(1-(q*q).sum(1))[:,None]);
  o=minimize_scalar(e,bounds=(.001,3.4),method='bounded');print('C2',name,o.x,o.fun);L.append((o.fun,f,o.x));
 _,f,a=min(L,key=lambda p:p[0]);return f,a,reg(D(f(u,a)),v)

def get():
 a,w=fit_a();x=np.r_[tr['x'],s['test']['x']];z=invert(x,a,w); print('range Z',z.min(),z.max());f,a,w=fit_b(); b=D(f(z,a))@w; f,a,w=fit_c(); c=D(f(z,a))@w; p=cyc(z);v=D(p)@reg(D(cyc(C['u'])),C['v1']);return x,z,b,c,p,v

def polys(x,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.prod(x[:,ix],1) for j in range(1,d+1) for ix in combinations_with_replacement(range(x.shape[1]),j)])

def scores(K,regu):
 vals=[]
 for i,j in [(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(52),j),j) for j in np.array_split(np.random.default_rng(2).permutation(52),4)]:
  p=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+regu*np.eye(len(i)),Y[i]);err=np.abs(p-Y[j]);vals.append([err[:,0].mean(),err[:,1:].mean()]);
 return np.array(vals)
if __name__=='__main__':
 import sys
 sys.path.insert(0,'/holdout_task/_frozen_nodes');from node_01 import _kernel
 x,z,b,c,p,v=get();np.savez('/tmp/data21.npz',x=x,z=z,b=b,c=c,p=p,v=v,Y=Y)
 combos={'x':x,'z':z,'b':b,'c':c,'v':v,'zb':np.c_[z,b],'bc':np.c_[b,c],'zbcv':np.c_[z,b,c,v],'zbc':np.c_[z,b,c],'bv':np.c_[b,v],'cv':np.c_[c,v],'zcv':np.c_[z,c,v],'cosz':np.cos(2.55*z),'cosz+p+c':np.c_[np.cos(2.55*z),p,c]}; best=[]
 for name,f in combos.items():
  f=(f-f[:52].mean(0))/np.maximum(f[:52].std(0),1e-5)
  for scale in [.35,.65,1,1.7,3]:
   G=_kernel(scale*f,scale*f);q=scores(G,.004);best.append((max(q[0]),q[0],q[1:].mean(0),name,scale,'frozen'))
  g=f@f.T/f.shape[1]; dd=cdist(f,f,'sqeuclidean')/f.shape[1];
  for deg in [1,2,3]:
   for l in [0,.1,.5,2]:
    G=(1+g)**deg+(np.exp(-l*dd) if l else 0)
    for ridge in [.0001,.005,.05]:
     q=scores(G,ridge);best.append((max(q[0]),q[0],q[1:].mean(0),name,deg,(l,ridge)))
 for z0 in sorted(best,key=lambda k:k[0])[:12]:print('joint',z0)
 for ix in [0,1]:
  for z0 in sorted(best,key=lambda k:k[1][ix])[:6]:print('head',ix,z0)
