exec(open('/tmp/wall_cv.py').read().split('for walls in itertools.product')[0])
import downstream_solver as dn
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
axis=dn._axis
counts=[s['nx'],s['ny']]
def responses(p):
    def custom(k,v,h,r,wall):
        a=wall
        seam,lo,hi=p[3*a:3*a+3]
        base=axis(k,v,h,r,'reflect'); periodic=axis(k,v,h,r,'periodic')
        ii=np.array([0,len(k)-1]); loss=csc_matrix((np.array([lo,hi])/h,(ii,ii)),shape=base.shape)
        return base+seam*(periodic-base)+loss
    dn._axis=custom
    try: return dn._responses(s,(0,1))[0]
    finally: dn._axis=axis

def fit(p,train): return test(responses(p),train,train)[0]**2
zero=np.zeros(6)
for kind,idx,lim in [('seams',[0,3],[(0,1)]*2),('leak',[1,2,4,5],[(-.15,.25)]*4),('joint',list(range(6)),[(0,1),(-.15,.25),(-.15,.25)]*2)]:
    def expand(q):
        p=zero.copy(); p[idx]=q; return p
    def learn(train):
        o=minimize(lambda q:fit(expand(q),train),zero[idx],method='L-BFGS-B',bounds=lim,options={'ftol':1e-12,'gtol':1e-8}); return expand(o.x)
    p=learn(ys); print(kind,'params',np.round(p,5),'fit',round(fit(p,ys)**.5,7))
    for groupset in (sh,groups//4):
        er=[]; base=[]
        for g in np.unique(groupset):
            train=groupset!=g; val=~train; q=learn(train)
            er.append(test(responses(q),train,val)[0]); base.append(test(responses(zero),train,val)[0]);
        print('fold rms/worst',np.round([np.sqrt(np.mean(np.square(er))),max(er)],7),'reflect',np.round([np.sqrt(np.mean(np.square(base))),max(base)],7),'folds',np.round(er,6))
