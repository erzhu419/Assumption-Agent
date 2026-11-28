exec(open('/tmp/inspect_node01.py').read().split("print('times'")[0])
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar

def calibrated(x, key, fn, bounds):
    design = lambda x,k: F(x,x,fn(k,x))
    v = c[key]
    def loss(k):
        A=design(u,k); return np.mean((A@fit(A,v)-v)**2)
    k=minimize_scalar(loss,bounds=bounds,method='bounded').x
    A=design(u,k)
    print('panel',key,k,loss(k)**.5,np.mean(loo(A,v)))
    return design(x,k)@fit(A,v)
a=calibrated(q,'v1',lambda k,x:np.arctan(k*x),(.8,4))
b=calibrated(q,'v2',lambda k,x:np.log1p(k*x*x),(.1,1))
poly=lambda x,d: np.column_stack([np.prod(x[:,ix],axis=1) for ix in combinations_with_replacement(range(x.shape[1]),d)])
base=F(q,q,poly(q,2)); co=fit(base,y); M=np.zeros((4,4))
for k,(i,j) in enumerate(combinations_with_replacement(range(4),2)): M[i,j]=M[j,i]=co[5+k]/(1+(i!=j))
print('response quad intercept / linear:',co[:5], 'matrix',M,'eigen',np.linalg.eigh(M)); print('residual quantiles',np.quantile(y-base@co,[0,.1,.5,.9,1]))
blocks={'q':q,'a':a,'b':b,'sat':np.arctan(2.3*q),'log':np.log1p(.36*q*q),'sin':np.sin(1.7*q),'quartic':q**4,'cube':q**3,'rat':q/(1+q*q)}
extras={}
for name,x in blocks.items(): extras[name]=x; extras[name+'^2']=poly(x,2)
for k,l in [('q','a'),('q','b'),('a','b'),('q','sat'),('q','log'),('sat','log'),('q','sin'),('q','rat')]:
    x,w=blocks[k],blocks[l]; extras[k+'*'+l]=(x[:,:,None]*w[:,None,:]).reshape(len(q),-1)
for k,x in [('q',q),('a',a),('b',b)]:
    for sg in [1,-1]:
        D=np.column_stack([x[:,i]+sg*x[:,j] for i in range(4) for j in range(i+1,4)])
        for tag,h in [('tanh',lambda d:np.tanh(1.7*d)),('sin',lambda d:np.sin(1.7*d)),('cos',lambda d:np.cos(1.7*d)),('rat',lambda d:1/(1+d*d))]: extras[f'{k}:{sg}:{tag}']=h(D)
for j,x in enumerate([q,a,b]):
    extras[f'p3_{j}']=poly(x,3); extras[f'p4_{j}']=poly(x,4)
N=len(y); ids=np.arange(N); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(N),6)]
def err(A,Y):
    return [np.mean(abs(A[v]@fit(A[t],Y[t])-Y[v])) for t,v in splits]
R=[]
for k,x in extras.items():
    es=err(F(q,base[:,1:],x),y); R.append((np.mean(es[1:]),max(es[1:]),es[0],k))
print('extra block fits avg/worst/public:',*sorted(R)[:16],sep='\n'); print('best public',sorted(R,key=lambda x:x[2])[:6]); print('trajectory quadratic splits',err(base,z))
D=np.column_stack(list(extras.values())); names=[f'{k}:{i}' for k,x in extras.items() for i in range(x.shape[1])]
for use in ['all','mechanistic']:
    ix=np.arange(D.shape[1]) if use=='all' else np.array([i for i,k in enumerate(names) if k.startswith(('q*a','q*b','a*b','a^2','b^2'))]); C=D[:,ix]; scores=[]
    for t,v in splits:
        B=base[t]; r=y[t]-B@fit(B,y[t]); A=C[t]-B@fit(B,C[t]); chosen=[]; norms=np.linalg.norm(A,axis=0); norms[norms<1.e-8]=np.inf; A=A/norms
        yh=[]
        for step in range(14):
            j=np.argmax(abs(A.T@r)); chosen.append(j)
            g=F(q,base[:,1:],C[:,chosen]); cf=fit(g[t],y[t]); r=y[t]-g[t]@cf; yh.append(np.mean(abs(g[v]@cf-y[v])))
        scores.append(yh)
        if len(scores)==1: print('public sparse selections',use,[names[ix[j]] for j in chosen])
    print('sparse',use,'public/mean/worst',np.array([scores[0],np.mean(scores[1:],axis=0),np.max(scores[1:],axis=0)]))
