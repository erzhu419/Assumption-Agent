from graph import *
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from itertools import combinations
np.set_printoptions(precision=5,suppress=True)
B=s['panel_b'];u=B['u'];v=B['v'];uc=c['u'];vc=c['v2'];
def prep(x,a,mode):
 if mode=='sine':return x-a[0]*np.sin(a[1]*x)
 if mode=='sineshift':return x-a[0]*np.sin(a[1]*x+a[2])
 if mode=='tanhshift':return x-a[0]*np.tanh(a[1]*x+a[2])
 if mode=='sineshiftquad':return x-a[0]*np.sin(a[1]*x+a[2])+a[3]*x*x
 if mode=='tanhshiftquad':return x-a[0]*np.tanh(a[1]*x+a[2])+a[3]*x*x
 if mode=='satshift':return x-a[0]*(x+a[2])/(1+a[1]*(x+a[2])**2)
 if mode=='satshiftquad':return x-a[0]*(x+a[2])/(1+a[1]*(x+a[2])**2)+a[3]*x*x
 if mode=='c':return x+a[0]*x*x+a[1]*x**3
 if mode=='c4':return x+a[0]*x*x+a[1]*x**3+a[2]*x**4
 if mode=='cw':return x+a*x*x
for mode,init in [('sine',[.4,2.85]),('sineshift',[.4,2.85,0]),('tanhshift',[.74,1.44,0]),('sineshiftquad',[.4,2.85,0,0]),('tanhshiftquad',[.74,1.44,0,0]),('satshift',[-1.2,.8,0]),('satshiftquad',[-1.2,.8,0,0]),('c',[.45,0]),('c4',[.45,0,0]),('cw',[.45]*4)]:
 a,b=(uc,vc) if mode.startswith('c') else (u,v)
 def p(x,g):return I(prep(x,g,mode))
 def res(g):
  d=p(a,g);return (d@ls(d,b)-b).ravel()
 f=least_squares(res,init,max_nfev=140);print('PANEL',mode,f.x,np.mean(abs(f.fun))); k=ls(p(a,f.x),b);ops[mode]=lambda x,mode=mode,g=f.x,k=k:I(prep(x,g,mode))@k

def cross(a,b):return np.column_stack([np.sum(a*np.roll(b,k,axis=1),axis=1) for k in range(4)])
def reg(d,t,ids,alpha):
 if not alpha:return d@ls(d[ids],t[ids])
 sd=np.maximum(d[ids].std(0),1e-6);sd[0]=1; a=d[ids]/sd; reg=alpha*np.eye(a.shape[1]);reg[0,0]=0;return (d/sd)@np.linalg.solve(a.T@a+reg,a.T@t[ids]);
def sinfit(d,ids,fn,plus=None):
 d=np.asarray(d);ds=d[ids]; y0=y[ids];plus=d if plus is None else plus;ps=plus[ids]; beta=ls(ds,y0);b2=ls(ps,y0);k=len(beta)
 def pred(p,d,ps):return d@p[:k]+p[-1]*fn(ps@p[k:-1]);
 fs=[least_squares(lambda p:pred(p,ds,ps)-y0,np.r_[beta,b2*scale,amp],max_nfev=125) for scale,amp in [(.65,.5),(1.,-.5),(2.,.5),(.65,-.5)]];f=min(fs,key=lambda f:f.cost); return pred(f.x,d,plus)
if __name__=='__main__':
 ids=np.arange(52); cnt=72;m=tr[ids].mean(0);V=np.linalg.svd(tr[ids]-m,full_matrices=False)[2][:3];T=(tr-m)@V.T;qoff=ls(I(pair(uc)),c['v1'])[0];rr=[]
 for bn in ('sin','sine','sat','tanhshiftquad','satshiftquad'):
  w=ops[bn](z); q=qop(w)+qoff
  for cn,r in [('w',ops['c'](w)),('z',ops['c'](z)),('q',ops['c'](q))]:
   base=np.c_[z,q,r]; blocks=[z,w,q,r]; cr=[cross(a,b) for a,b in combinations(blocks,2)];
   for name,D,a in [('cyc',np.c_[base,*cr],.03),('zq',np.c_[base,outer(z,q)],.001),('both',np.c_[base,outer(z,q),*cr],.05),('full',np.c_[base,outer(z,q),outer(w,q),*cr],.25)]:
    df=I(D);tp=reg(df,T,ids,a);dt=np.abs(tp[52:72]-T[52:]).mean(0)
    if dt.sum()>.075:continue
    yt=[]
    for f,fn in [('sin',np.sin),('tanh',np.tanh),('rat',lambda h:h/(1+h*h))]:
     d=I(np.c_[z,q,tp]);dh=I(np.c_[z,q,np.r_[T,tp[72:]]]);
     for method in ['pred','mixed']:
      h=sinfit(d if method=='pred' else np.r_[dh[:52],d[52:]],ids,fn);e=np.abs(h[52:72]-y[52:]).mean();rr.append((e,bn,cn,name,f,method,dt.tolist()))
 print('JOINT',*sorted(rr)[:25],sep='\n')
