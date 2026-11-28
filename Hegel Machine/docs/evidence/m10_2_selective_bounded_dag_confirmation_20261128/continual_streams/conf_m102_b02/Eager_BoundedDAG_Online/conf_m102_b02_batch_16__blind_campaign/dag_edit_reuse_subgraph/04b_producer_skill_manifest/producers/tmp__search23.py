import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve, solve
from scipy.special import erf
from base23 import coordinates, D, cyc, head

def banks(z,B,C,E):
    b,c,e=B(z),C(z),E(z)
    q={'z':z,'b':b,'c':c,'e':e}
    for nm,x in list(q.items()):
        if nm!='z':
            for name,func in [('b',B),('c',C),('e',E)]:q[name+nm]=func(x)
    for nm,x in [('z',z),('b',b),('c',c),('e',e)]:
        for nn,y in [('z',z),('b',b),('c',c),('e',e)]:
            if nm<nn:
                for a in [-1.,1.]:
                    x1=x+a*y; k=f'{nm}{a:+}{nn}';q[k]=x1
                    for name,func in [('b',B),('c',C),('e',E)]:q[name+k]=func(x1)
    g={}
    for nm,x in q.items():
        i,j=np.triu_indices(x.shape[1],1)
        g[nm+'lin']=x
        g[nm+'quad']=np.c_[x*x,x[:,i]*x[:,j]]
        g[nm+'cyc']=cyc(x)
        for a in [.65,1.25,2.15]:
            for k,h in [('tanh',np.tanh(a*x)),('erf',erf(a*x)),('sin',np.sin(a*x)),('rat',x/(1+(a*x)**2)),('th',np.sign(x)*np.maximum(abs(x)-.22,0)**1.25)]:
                g[nm+k+str(a)]=h
                if a==1.25:g[nm+k+'2']=h*h
    for i,(nm,x) in enumerate(list(q.items())[:13]):
        for nn,y in list(q.items())[i+1:13]:
            g[nm+'*'+nn]=(x[:,:,None]*y[:,None,:]).reshape(len(z),-1)
    return g

def covs(g,n):
    groups=[];names=[]
    for name,F in g.items():
        F=F-F[:n].mean(0); scale=np.maximum(F[:n].std(0),.008); F=F/np.sqrt(scale); K=F@F.T
        K/=max(np.mean(K.diagonal()[:n]),1e-10);groups.append(K);names.append(name)
    # Broad smooth components as a fallback.
    for name in ['z','b','c','e','bz','cb','eb','ez','cz','bz-1.0e']:
        if name+'lin' not in g:continue
        x=g[name+'lin']; x=(x-x[:n].mean(0))/np.maximum(x[:n].std(0),.01)
        dot=x@x.T/x.shape[1]; dist=np.maximum(dot.diagonal()[:,None]+dot.diagonal()-2*dot,0)
        for a in [.08,.3,1.2]:groups.append(np.exp(-a*dist));names.append(name+'rbf'+str(a))
    return names,np.array(groups)

def regress(K,Y,verbose=False,names=None):
    n=len(Y); avg=Y.mean(0); sd=np.maximum(Y.std(0),1e-9); Y=(Y-avg)/sd
    flat=np.ascontiguousarray(K[:,:n,:n].reshape(len(K),-1)); cross=np.ascontiguousarray(K[:,n:,:n].reshape(len(K),-1)); eye=np.eye(n); ps=[]
    for t in Y.T:
        def obj(w):
            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye
            c=cho_factor(V,check_finite=False); a=cho_solve(c,t,check_finite=False); W=cho_solve(c,eye,check_finite=False)-a[:,None]*a[None,:]
            return np.log(c[0].diagonal()).sum()+.5*(t@a), np.r_[.5*(flat@W.ravel()),.5*W.trace()]
        pp=[]
        for penalty in [0.,2.,8.]:
            def fn(w):
                l,d=obj(w);return l+penalty*np.sum(w[:-1]),d+np.r_[np.full(len(K),penalty),0.]
            init=np.r_[np.full(len(K),1./len(K)),.015]; w=minimize(fn,init,jac=True,method='L-BFGS-B',bounds=[(0,15.)]*len(K)+[(.000025,1.)],options={'maxiter':180,'ftol':4e-8}).x
            if verbose and penalty==2.:print('groups',[(names[i],round(w[i],3)) for i in np.argsort(w[:-1])[-12:][::-1] if w[i]>.0001], 'noise',w[-1],flush=True)
            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye; cross1=(w[:-1]@cross).reshape(-1,n); a=solve(V,t,assume_a='pos'); pp.append(np.r_[t-w[-1]*a,cross1@a]);
        ps.append(np.mean(pp,axis=0))
    return np.column_stack(ps)*sd+avg

if __name__=='__main__':
    from probe23 import S
    from model23 import get
    z,B,C,E=coordinates(S); _,_,_,f,Y,_=get(S); u=S['panel_a']['u']; v=S['panel_a']['v']; print('panels',S.keys(),[(k,{j:(a.shape,np.round(np.array([a.min(),a.max()]),4)) for j,a in S[k].items()}) for k in ['panel_a','panel_b','panel_c']]);g=banks(z,B,C,E)
    for tr,va in [(np.arange(52),np.arange(52,72)),(np.arange(72)[np.arange(72)%4!=0],np.arange(0,72,4))]:
        n=len(tr); ids=np.r_[tr,va]; names,K=covs({name:a[ids] for name,a in g.items()},n); P=regress(K,np.c_[Y[:,0],Y[:,1:]@np.linalg.svd(f,full_matrices=False)[2].T][tr],True,names)[n:]; err=Y[va,1:]@np.linalg.svd(f,full_matrices=False)[2].T-P[:,1:]; print('MAE',np.mean(abs(P[:,0]-Y[va,0])),np.mean(abs(err@np.linalg.svd(f,full_matrices=False)[2]@f.T)),flush=True)
