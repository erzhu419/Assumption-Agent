import sys,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task')
from batch_solver import _panels,_aff,_quad,_ols,solve_batch

def _gram(f,n):
    f=f-f[:n].mean(0);f=f/np.maximum(f[:n].std(0),.04)
    return f@f.T/f.shape[1]

def parts(z,y,tr,family,link):
    x=z['x'];n=len(y);d=z['bi1'];e=z['i2j']; A=_aff(x); Q=_quad(x)
    q=_ols(Q[:n],y); H=np.zeros((4,4));H[np.triu_indices(4)]=q[5:]; H=(H+H.T)*.5
    _,V=np.linalg.eigh(H); p=x@V[:,[0,-1]]
    B=_ols(Q[:n],tr); w=(Q@B); u,ss,v=np.linalg.svd(tr,full_matrices=False); w=w@v[:2].T
    if family==0: vs=[x,p,d,e,w]
    elif family==1: vs=[x,p,np.c_[d,e],w]
    elif family==2: vs=[x,p,w]
    else: vs=[x,p,np.c_[z['bi1'],z['bi2']],w]
    Ks=[np.ones((len(x),len(x))),_gram(x,n),_gram(Q[:,5:],n)]
    for k,b in enumerate(vs):
        g=_gram(b,n); plus=_gram(_quad(b)[:,1+b.shape[1]:],n)
        if k:
            Ks.extend([g,plus,g*Ks[1]])
        if k in (0,1,len(vs)-1):
            dn=(b-b[:n].mean(0))/np.maximum(b[:n].std(0),.04); dist=cdist(dn,dn)/np.sqrt(b.shape[1]);
            for l in (.7,1.6,3.0):
                r=dist*np.sqrt(5)/l;Ks.append((1+r+r*r/3)*np.exp(-r))
    K=np.array(Ks); Kr=K[:,:n,:n].copy();Y=y if link==0 else np.sinh(y/link)*link
    G=np.concatenate([Kr,np.eye(n)[None]])
    g=G.reshape(len(G),-1); start=np.r_[.1,.05,.05,np.full(len(G)-4,.004),.0002]; l0=np.log(start)
    def loss(l):
        s=np.exp(l);c=cho_factor((s@g).reshape(n,n));a=cho_solve(c,Y);ic=cho_solve(c,np.eye(n));
        return .5*Y@a+np.log(np.diag(c[0])).sum()+.025*np.sum((l-l0)**2),.5*s*(g@(ic-np.outer(a,a)).ravel())+.05*(l-l0)
    o=minimize(loss,l0,jac=True,method='L-BFGS-B',bounds=[(-18,4)]*len(l0),options={'maxiter':85});s=np.exp(o.x);a=cho_solve(cho_factor((s@g).reshape(n,n)),Y)
    h=np.einsum('k,kij,j->i',s[:-1],K[:,n:,:n],a)
    return h if link==0 else np.arcsinh(h/link)*link

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];x,y,tr=t['x'],t['y'],t['traj']; I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72],np.arange(56)]
 for i in I:
    j=np.setdiff1d(np.arange(len(x)),i);z=_panels(s,x[np.r_[i,j]]); d={**s,'train':{k:t[k][i] for k in ('x','y','traj')},'test':{'x':x[j]}}; b=solve_batch(d)['y_hat'];
    P=np.array([parts(z,y[i],tr[i],f,l) for f in range(4) for l in (0,1.,1.7)]); print('GP',len(i),j[0],np.mean(abs(P-y[j]),axis=1).round(5),np.mean(abs(P.mean(0)-y[j])),np.mean(abs((.25*b+.75*P.mean(0))-y[j])),flush=True)
 print('projections',np.linalg.lstsq(_quad(x),np.c_[y,tr[:,2],tr[:,-1]],rcond=None)[0].round(3))
