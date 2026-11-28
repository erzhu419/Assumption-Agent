import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import *
from scipy.optimize import nnls
from scipy.spatial.distance import cdist

def ard(w,Y,n,kind,init):
    w=(w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-8); Y=np.asarray(Y); m=Y.mean(0); sc=np.maximum(Y.std(0),1e-8); Y=(Y-m)/sc
    d=((w[:n,None,:]-w[None,:n,:])**2).transpose(2,0,1).copy(); dc=((w[n:,None,:]-w[None,:n,:])**2).transpose(2,0,1).copy(); p=len(d); I=np.eye(n); G=1+w[:n]@w[:n].T/p; Gc=1+w[n:]@w[:n].T/p
    def ker(h,dist,G,train):
        a=np.exp(h); s=np.einsum('i,ijk->jk',a[:p],dist)
        if kind==0: k=np.exp(-s); grad=-k
        if kind==1: r=np.sqrt(3*s+1e-30); v=np.exp(-r); k=(1+r)*v; grad=-1.5*v
        if kind==2: r=np.sqrt(5*s+1e-30); v=np.exp(-r); k=(1+r+r*r/3)*v; grad=-(5/6)*(1+r)*v
        if kind==3: k=(1+s)**-2; grad=-2*(1+s)**-3
        C=a[p]*k+a[p+1]*G+a[p+2]*G**2
        if not train:return C
        return C+a[-1]*I,[*(a[:p,None,None]*dist*(a[p]*grad)[None,:,:]),a[p]*k,a[p+1]*G,a[p+2]*G**2,a[-1]*I]
    def loss(h):
        C,J=ker(h,d,G,True); L=cho_factor(C); A=cho_solve(L,Y); B=cho_solve(L,I)-A@A.T/Y.shape[1]; delta=h[:p]+3.0; return .5*np.sum(A*Y)/Y.shape[1]+np.log(L[0].diagonal()).sum()+.002*sum(delta**2), np.einsum('ijk,jk->i',J,B)*.5+np.r_[.004*delta,[0]*4]
    opt=minimize(loss,np.r_[np.full(p,init),0,-4,-4,-7],jac=True,method='L-BFGS-B',bounds=[(-11,4)]*p+[(-8,9),(-12,6),(-12,6),(-14,-2)],options={'maxiter':130})
    C,_=ker(opt.x,d,G,True); L=cho_factor(C); A=cho_solve(L,Y); loo=Y-A/cho_solve(L,I).diagonal()[:,None]; return (m+ker(opt.x,dc,Gc,False)@A*sc),m+loo*sc,np.exp(opt.x),opt.fun

def fixed(w,Y,n,deg,ls,l):
    w=(w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-9); D=cdist(w,w,'sqeuclidean')/w.shape[1]; G=1+w@w.T/w.shape[1]; K=G**deg+np.exp(-D/ls)*3; L=cho_factor(K[:n,:n]+l*np.eye(n)); a=cho_solve(L,Y); return K[n:,:n]@a,Y-a/cho_solve(L,np.eye(n)).diagonal()

def sbranch(panel,x,form):
    u,v=[panel[k] for k in ('u','v')]; a=affine(u)
    if form==0: fn=lambda u,h:np.c_[affine(u),np.sin(h[0]*u)]
    if form==1: fn=lambda u,h:np.c_[affine(u),np.tanh(h[0]*u),np.sin(h[1]*u)]
    if form==2: fn=lambda u,h:np.c_[affine(u),np.cos(h[0]*u)]
    def r(h):
        A=fn(u,h);return (A@fit(A,v)-v).ravel()
    opt=min((least_squares(r,ini,bounds=(.15,7),max_nfev=120) for ini in [(1,1),(2,2),(3.5,1.25),(1.25,3.5),(4.5,2.5)]),key=lambda o:o.cost)
    return fn(x,opt.x)@fit(fn(u,opt.x),v)

def examine(s,z,b,t,y,jj,kk,extended=True):
    idx=np.r_[jj,kk]; n=len(jj); z,b=z[idx],b[idx]; t=t[idx]; y=y[jj]; U,sv,V=np.linalg.svd(t[:n],full_matrices=False); pcs=t[:n]@V[:3].T
    B=sbranch(s['panel_b'],z,0); C=sbranch({'u':s['panel_c']['u'],'v':s['panel_c']['v1']},z,1); D=sbranch({'u':s['panel_c']['u'],'v':s['panel_c']['v2']},z,2)
    W={'z':z,'b':b,'B':B,'zb':np.c_[z,b],'zB':np.c_[z,B],'bC':np.c_[b,C],'bD':np.c_[b,D],'bCD':np.c_[b,C,D],'zbCD':np.c_[z,b,C,D]}; out={}
    for name,w in W.items():
        for k in [0,2,3]:
            for yi in [0,1,2]:
                target=np.c_[y,pcs[:,1:]] if yi==1 else np.c_[y,pcs] if yi==2 else y[:,None]; pr,lo,h,c=ard(w,target,n,k,-3)
                out[(name,k,yi)]=(pr[:,0],lo[:,0],c,h)
        for degree in [1,2,3]:
            for ls in [2,6,16]:
                for l in [.0001,.001,.01]:
                    pr,lo=fixed(w,y,n,degree,ls,l);out[(name,degree,ls,l)]=(pr,lo,0,[])
    return out

if __name__=='__main__':
    from probe import s
    z,b,n=features(s); y=s['train']['y'];t=s['train']['traj'];
    allout=[]
    for i in [0,10,17,25,42]:
        jj=np.arange(52) if not i else np.random.default_rng(i).permutation(72)[:52]; kk=np.setdiff1d(np.arange(72),jj); out=examine(s,z,b,t,y,jj,kk); allout.append((jj,kk,out));
        print('split',i,flush=True)
        orders=lambda which: sorted(out,key=lambda k:np.mean(abs(out[k][which]-y[[kk,jj][which]])))
        for which in [0,1]:
            for k in orders(which)[:8]:print(k,round(np.mean(abs(out[k][0]-y[kk])),5),round(np.mean(abs(out[k][1]-y[jj])),5),np.round(out[k][3],3),flush=True)
        for typ in [0,1,2]:
            keys=[k for k in orders(1) if not typ or len(k)==typ+2];
            for count in [3,10,25]:print('avg',typ,count,np.mean(abs(np.mean([out[k][0] for k in keys[:count]],0)-y[kk])),flush=True)
    import pickle
    pickle.dump(allout,open('/tmp/ard.pkl','wb')); out=allout[0][-1]
    for k in sorted(out,key=lambda k:np.mean([np.mean(abs(o[k][0]-y[kk])) for j,kk,o in allout]))[:15]:print('aggregate', k,np.round([np.mean(abs(o[k][0]-y[kk])) for j,kk,o in allout],5))
