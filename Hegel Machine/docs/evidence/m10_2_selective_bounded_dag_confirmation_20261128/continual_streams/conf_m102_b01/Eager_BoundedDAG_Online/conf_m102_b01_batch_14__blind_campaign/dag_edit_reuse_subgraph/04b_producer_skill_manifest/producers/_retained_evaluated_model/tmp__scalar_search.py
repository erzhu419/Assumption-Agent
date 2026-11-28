import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

def aff(x): return np.c_[np.ones(len(x)),x]
def ls(a,y): return np.linalg.lstsq(a,y,rcond=1e-8)[0]
def quad(x):
    i,j=np.triu_indices(x.shape[1]); return x[:,i]*x[:,j]
def transform(t,k):
    if k=='a': return np.arctan(t)
    if k=='s': return np.arcsinh(t)
    if k=='t': return np.tanh(t)
    if k=='i': return np.sinh(np.arcsinh(t)/3)
    return t/np.sqrt(1+t*t)

def designs(z,c1,c2):
    yield 'C',aff(np.c_[z,c1,c2])
    for k in ['a','s','t','i','r']:
        for g in [1.1,1.8,2.6,3.4]:
            v=transform(g*z,k); q=quad(v)
            yield f'{k}/{g}/q',aff(np.c_[v,q]); yield f'{k}/{g}/c',aff(np.c_[v,q,c1,c2]); yield f'{k}/{g}/c2',aff(np.c_[v,c1,c2,quad(c1)]); yield f'{k}/{g}/d',aff(np.c_[v,q,c2]); yield f'{k}/{g}/b',aff(np.c_[v,q,c1]); yield f'{k}/{g}/j',aff(np.c_[v,q,np.sin(2.6*z),np.cos(2.6*z)])
    for name,v in [('c1',c1),('c2',c2),('z',z)]:
        for k in ['a','t','s']:
            for g in [0.8,1.6,2.6]:
                f=transform(g*v,k); yield f'{name}/{k}/{g}',aff(np.c_[f,quad(f),z]); yield f'{name}/{k}/{g}/f',aff(np.c_[f,quad(f),z,c1,c2])

def ridge(a,y,reg):
    mu=a[:,1:].mean(0); scale=np.maximum(a[:,1:].std(0),1e-6); d=(a[:,1:]-mu)/scale
    coef=np.linalg.solve(d.T@d+reg*np.eye(d.shape[1]),d.T@(y-y.mean()))/scale
    return np.r_[y.mean()-mu@coef,coef]

def proj(z,c1,c2,y,kind,base,mode):
    n=len(y); ii=np.arange(n); a=dict((k,x) for k,x in [('C',aff(np.c_[z,c1,c2])),('q',aff(np.c_[np.arctan(2.15*z),quad(np.arctan(2.15*z))])),('p',aff(np.c_[z,quad(z)])),('v',aff(np.c_[z,c1,c2,quad(np.arctan(1.8*z))]))])[base]; h={'z':z,'b':c1,'c':c2,'1':z[:,1:2],'i':np.arctan(1.8*z)}[mode]; mu=h[ii].mean(0); sd=np.maximum(h[ii].std(0),1e-6); h=aff((h-mu)/sd)
    f={'sin':np.sin,'cos':np.cos,'gauss':lambda x:np.exp(-x*x),'tanh':np.tanh,'cube':lambda x:x**3,'r':lambda x:x/(1+x*x),'erf':erf,'soft':lambda x:np.sqrt(1+x*x)}[kind]
    def des(p):return np.c_[a,f(h@p)]
    def fun(p):
        d=des(p)[:n]; return np.r_[d@ls(d,y)-y, .0004*p]
    raw=ls(h[:n],y); raw/=max(np.linalg.norm(raw[1:]),.05)
    target=y-a[:n]@ls(a[:n],y); r=ls(h[:n],target); r/=max(np.linalg.norm(r[1:]),.02)
    rng=np.random.default_rng(139); starts=[raw*g+np.r_[shift,np.zeros(h.shape[1]-1)] for g in [.35,1.,2.2] for shift in [0.,.75,-.75]]+[r,.5*r]+[rng.normal(0,.6,h.shape[1]) for k in range(4)]
    best=min((least_squares(fun,p,max_nfev=60,bounds=(-35,35)) for p in starts),key=lambda p:p.cost); d=des(best.x); return d[n:]@ls(d[:n],y)

def nonlinear(z,c1,c2,y,mode,kind,rank):
    h={'z':z,'c':c1,'d':c2,'a':np.arctan(1.6*z),'u':np.c_[z,c1,c2]}[mode]; n=len(y); h=aff((h-h[:n].mean(0))/np.maximum(h[:n].std(0),1e-6)); m=h.shape[1]; a=aff(np.c_[z,c1,c2]) if mode=='u' else aff(z); f={'t':np.tanh,'a':np.arctan,'s':np.sin,'r':lambda x:x/np.sqrt(1+x*x),'g':lambda x:np.exp(-x*x)}[kind];
    def design(p): return np.c_[a,f(h@p.reshape(m,rank))]
    def loss(p):
        d=design(p)[:n]; return np.r_[d@ridge(d,y,.0003)-y,.0015*p]
    rng=np.random.default_rng(14); r=ls(h[:n],y); r/=max(np.linalg.norm(r[1:]),.05)
    starts=[]
    for k in range(12):
        p=rng.normal(0,[.35,.8,1.6][k%3],(m,rank)); p[:,0]=r*[.5,1.4,2.5][k%3]; p[0,:]+=rng.uniform(-1,1,rank); starts.append(p.ravel())
    best=min((least_squares(loss,p,max_nfev=85,bounds=(-40,40)) for p in starts),key=lambda p:p.cost); d=design(best.x); return d[n:]@ridge(d[:n],y,.0003)

if __name__=='__main__':
    exec(open('/tmp/compose.py').read().split('for j in [')[0].replace('idx=np.arange(72)<52; R=np.c_[Y,T];','idx=np.arange(72)<52; R=np.c_[Y,T];'))
    z=V['b']; c1=C(z); c2=D(z); folds=[np.random.default_rng(31+i).permutation(52)[:13] for i in range(3)]; results=[]; preds={}
    for key,d in designs(z,c1,c2):
        for reg in [.0003,.025,.2,1]:
            out=[]
            for i in folds:
                j=np.setdiff1d(np.arange(52),i); out.append(np.mean(abs(d[i]@ridge(d[j],Y[j],reg)-Y[i])))
            pr=d[52:]@ridge(d[:52],Y[:52],reg); name=('lin',key,reg); results.append((np.mean(out),max(out),np.mean(abs(pr-Y[52:])),name)); preds[name]=pr
    opts=[('proj',k,b,m) for k in ['sin','gauss','tanh','cube','soft'] for b in ['C','q','v'] for m in ['z','b','1','i']]+[('nonlin',m,k,r) for m in ['z','b','c','a'] for k in ['t','s','r','g'] for r in [2,3]]
    opts=[o for o in opts if not(o[0]=='nonlin' and o[1]=='b')]
    for opt in opts:
        fn=proj if opt[0]=='proj' else nonlinear; p=fn(z,c1,c2,Y[:52],*opt[1:]); err=np.mean(abs(p-Y[52:])); preds[opt]=p; results.append((np.inf,np.inf,err,opt));
    top=sorted(results,key=lambda q:q[2]); print('TOP TAIL',top[:26],flush=True)
    selected={q[-1] for q in top[:24]}|{q[-1] for q in sorted(results)[:12]}; comp=[]
    for name in selected:
        o=[]
        for i in folds:
            j=np.setdiff1d(np.arange(52),i); iz=np.r_[j,i];
            if name[0]=='lin':
                d=dict(designs(z,c1,c2))[name[1]]; p=d[i]@ridge(d[j],Y[j],name[2]);
            else: p=(proj if name[0]=='proj' else nonlinear)(z[iz],c1[iz],c2[iz],Y[j],*name[1:]);
            o.append(p)
        e=[np.mean(abs(p-Y[ii])) for p,ii in zip(o,folds)]; comp.append((np.mean(e),max(e),np.mean(abs(preds[name]-Y[52:])),name,o))
    comp.sort(key=lambda x:x[0]); print('REPEATS',[x[:-1] for x in comp],flush=True)
    for count in [1,2,3,4,6,8]:
        subset=comp[:count]; p=np.mean([preds[x[3]] for x in subset],0); scores=[np.mean(abs(np.mean([x[4][j] for x in subset],0)-Y[i])) for j,i in enumerate(folds)]; print('ENSEMBLE',count,np.mean(abs(p-Y[52:])),scores,flush=True)
    import pickle; pickle.dump((comp,preds),open('/tmp/scalar_choices.pkl','wb'))
