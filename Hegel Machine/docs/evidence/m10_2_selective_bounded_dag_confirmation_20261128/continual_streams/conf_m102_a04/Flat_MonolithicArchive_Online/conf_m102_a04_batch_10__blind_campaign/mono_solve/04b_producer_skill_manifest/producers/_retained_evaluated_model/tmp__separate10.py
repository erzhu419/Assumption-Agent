exec(open('/tmp/audit10.py').read().split('for F,tar')[0])
from scipy.optimize import least_squares
T=s['eval_times']; diag=(g*h).sum(1); off=g[:,0]*h[:,1]; A=np.c_[g,diag]; B=np.c_[h,off]; FF=m.one(np.c_[A,B]);

def fit_modes(A,B,Y,t,style):
    n,p=A.shape; q=B.shape[1]
    def design(r,aa,bb):
        fast=-np.expm1(-r[0]*t)
        slow=-np.expm1(-r[1]*t)/r[1]
        if style==1: slow=slow+r[2]*t
        if style==2: slow=slow+r[2]*(t*t)
        return np.concatenate([aa[:,:,None]*fast,bb[:,:,None]*slow],axis=1).transpose(0,2,1).reshape(len(aa)*len(t),p+q)
    def loss(r):
        F=design(r,A,B);return F@m.lin(F,Y.ravel())-Y.ravel()
    fit=least_squares(loss,[1.8,.42]+([0.] if style else []), bounds=([.5,.025]+([-1.] if style else []),[3.,2]+([1.] if style else [])),gtol=1e-11)
    F=design(fit.x,A,B); coeff=m.lin(F,Y.ravel());return fit.x,lambda a,b:(design(fit.x,a,b)@coeff).reshape(len(a),len(t))

groups=[np.arange(52,72),np.argsort(np.linalg.norm(z,axis=1))[-20:],*(np.random.default_rng(i).permutation(n)[:20] for i in range(8))]
for style in (0,1,2):
    scores=[]
    for test in groups:
        fit=np.setdiff1d(np.arange(n),test); q,pred=fit_modes(A[fit],B[fit],Y[fit,1:],T,style); error=np.abs(pred(A[test],B[test])-Y[test,1:]);scores.append([error.mean(),error.mean(0).max(),np.mean(error[:,-1])]);
        if len(scores)==1: print('rates',style,q)
    print('mix',style,'public',scores[0],'worst',np.max(scores,axis=0),'mean',np.mean(scores,axis=0))
