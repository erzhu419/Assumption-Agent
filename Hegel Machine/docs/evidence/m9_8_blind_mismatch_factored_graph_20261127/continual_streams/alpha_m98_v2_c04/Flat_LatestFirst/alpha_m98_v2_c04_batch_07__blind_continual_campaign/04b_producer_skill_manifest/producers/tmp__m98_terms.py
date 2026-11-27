from m98_chamber import *
from itertools import product

def fit2(take, term, alignment, fixed, mult=1.):
    dd=alignment
    def design(p):
        a,b,c,e=np.exp(p)
        if fixed: c,e=a,b
        g,j=z(c),z(e); H=h-g; J=h-j
        u=H[:,None]; v=np.c_[J,dd]; w=np.c_[one,h,dd]
        name, placement=term
        if name!='none':
            extra={'sq':dd**2,'abs':abs(dd),'cube':dd**3,'tanh':np.tanh(dd),'ratio':dd/(1+dd**2),'prod':dd*J,'hist':j,'hist0':g,'hold':h*h,'fast':H*dd,'histdrift':dd*j,'holdclip':h/(1+h*h/12)}[name]
            if placement==0: u=np.c_[u,extra]
            elif placement==1: v=np.c_[v,extra]
            else: w=np.c_[w,extra]
        bs=[np.r_[-np.expm1(-t/a),1/a],np.r_[-np.expm1(-t/b),1/b],np.r_[t,1.]]
        return np.concatenate([f[:,None,:]*bt[None,:,None] for f,bt in zip((u,v,w),bs)],axis=2)
    p0=np.log([.47,1.2,.47,1.2]); A=design(p0); pp=np.r_[p0,np.linalg.lstsq(A[take].reshape(-1,A.shape[-1]),y[take].ravel(),rcond=None)[0]]
    def pred(p): return design(p[:4])@p[4:]
    lim=(np.r_[np.log([.25,.65,.15,.15]),np.full(len(pp)-4,-np.inf)],np.r_[np.log([.65,8.,2.,12.]),np.full(len(pp)-4,np.inf)])
    f=least_squares(lambda p:((pred(p)-y)[take]*np.r_[np.ones(len(t)),mult]).ravel(),pp,bounds=lim,loss='soft_l1',f_scale=.012,max_nfev=150)
    return pred(f.x)-y, f.x

def report(E):
    sl=[lab==l for l in np.unique(lab)]+[q['stratum']==s for s in np.unique(q['stratum'])]
    return np.array([np.median(abs(E[:,:-1])),np.median(abs(E[:,-1])),max(np.median(abs(E[ix,:-1])) for ix in sl),max(np.median(abs(E[ix,-1])) for ix in sl)])
xs=[d,th[2]+m._features(tr,q)[n:]@th[3:],th[2]+m._features(q,q)[:len(h)]@th[3:]]
rows=[]
for i in range(3):
 for fixed in (True,False):
    for term in [('none',0)]+list(product(('sq','abs','cube','tanh','ratio','prod','hist','hist0','hold','fast','histdrift','holdclip'),range(3))):
        if i and term[0] not in ('none','sq','abs','ratio','hist','prod'): continue
        E,p=fit2(np.ones(len(y),bool),term,xs[i],fixed); scores=report(E); rows.append((scores[0]+scores[1],i,fixed,term,scores,p))
for _,i,fixed,term,scores,p in sorted(rows,key=lambda x:x[0])[:20]+[r for r in rows if r[3][0]=='none']:
    cv=np.zeros_like(y)
    for l in np.unique(lab):
        ix=lab==l; E,_=fit2(~ix,term,xs[i],fixed); cv[ix]=E[ix]
    print(i,fixed,term,'fit',np.round(scores,4),'CV',np.round(report(cv),4),'times',np.round(np.exp(p[:4]),3),'coeff',np.round(p[4:],3),flush=True)
