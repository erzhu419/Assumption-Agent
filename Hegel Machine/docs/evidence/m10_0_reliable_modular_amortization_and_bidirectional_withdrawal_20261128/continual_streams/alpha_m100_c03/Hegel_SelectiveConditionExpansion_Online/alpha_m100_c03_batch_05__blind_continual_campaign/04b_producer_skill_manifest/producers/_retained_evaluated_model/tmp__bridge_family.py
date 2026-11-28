import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=c['true_channel_loads'],c['bridge_raw']; keep=np.max(abs(Y),axis=1)<5; X,Y=X[keep],Y[keep]; print('clean n',len(X),'excluded rows',np.flatnonzero(~keep)); print('axial rows',np.round(np.c_[X,Y][(X!=0).sum(1)<=1],5))
F={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'sin':np.sin,'rational':lambda t:t/(1+t*t),'sqrt':lambda t:t/np.sqrt(1+t*t)}
Q=[(i,j) for i in range(4) for j in range(i,4)]
def design(x,k,j,f,q,scope,offset):
    z=x[:,j:j+1] if scope=='diag' else x
    if offset: w,b=np.split(k,2)
    else: w,b=k,0
    sz=w*z+b
    return np.column_stack((np.ones(len(x)),x,*[x[:,a]*x[:,b] for a,b in q],F[f](sz)))
def fit(x,y,j,f,q,scope,offset=False):
    def mat(k,x=x):return design(x,k,j,f,q,scope,offset)
    def coef(a):return np.linalg.lstsq(a,y,rcond=None)[0]
    def res(k):
        a=mat(k); return a@coef(a)-y
    n=1 if scope=='diag' else 4
    lo=[.08]*n+([-1.5]*n if offset else []); hi=[2.6]*n+([1.5]*n if offset else [])
    ans=min((least_squares(res,[s]*n+([0]*n if offset else []),bounds=(lo,hi),max_nfev=65) for s in ([.45,.95,1.65] if f=='sin' else [.6,1.3])),key=lambda p:p.cost)
    return ans.x,coef(mat(ans.x))
rows=[]
for f,scope,qn,offset in itertools.product(F,['diag','all'],['none','diag','all'],[False,True]):
    if scope=='all' and offset:continue
    q=[] if qn=='none' else [t for t in Q if qn=='all' or t[0]==t[1]]
    R=[]; H=[]; P=[]
    for j in range(4):
        args=(j,f,q,scope,offset); k,p=fit(X,Y[:,j],*args); a=design(X,k,*args)
        # Tangent-design leverage includes optimized curvature/offset parameters.
        h=1e-5; tangent=np.column_stack([((design(X,k+h*e,*args)-design(X,k-h*e,*args))@p)/(2*h) for e in np.eye(len(k))]); u,s,_=np.linalg.svd(np.column_stack((a,tangent)),full_matrices=False); lev=(u[:,s>1e-8]**2).sum(1); r=Y[:,j]-a@p; R.append(r); H.append(r/np.maximum(1e-4,1-lev)); P.append((k,p));
    r,h=np.array(R),np.array(H); rows.append((np.max(np.mean(h*h,1)**.5),np.max(abs(h)),f,scope,qn,offset,np.mean(r*r,1)**.5,P))
for t in sorted(rows,key=lambda t:t[0])[:16]: print(t[:7]);
for t in sorted(rows,key=lambda t:t[0])[:3]:
    print('COEFS',t[2:6]);
    for k,p in t[-1]:print(np.round(k,6),np.round(p,6));
