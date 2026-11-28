from features import *
from scipy.optimize import least_squares
from itertools import product
from sys import path
path.insert(0,'/holdout_task')
from batch_solver import _gp

def readout(x,z,y,kind):
    aa=af(x); az=af(z); fq=pol(x,2)
    co=ls(fq,y); Q=np.zeros((4,4))
    for w,(i,j) in zip(co[5:],combinations_with_replacement(range(4),2)):Q[i,j]=Q[j,i]=w if i==j else w/2
    e,V=np.linalg.eigh(Q); a=V[:,-1]*np.sqrt(max(e[-1],.01)); b=V[:,0]*np.sqrt(max(-e[0],.01))
    u=a+b; v=a-b
    def sat(q,k):
        if k=='t':return np.tanh(q)
        if k=='s':return np.sin(q)
        if k=='r':return q/np.sqrt(1+q*q)
        return q
    def des(A,p):
        P=A@p[:5]; R=A@p[5:10]; X=A[:,1:]; C=[A]
        if kind[0]=='p':
            C.extend([sat(P,kind[1])*sat(R,kind[2])])
            if kind.endswith('d'): C.extend([sat(P,kind[1]),sat(R,kind[2])])
        elif kind[0]=='q':
            C.extend([sat(P,kind[1])**2,sat(R,kind[2])**2]);
            if kind.endswith('d'):C.extend([sat(P,kind[1]),sat(R,kind[2])])
        elif kind[0]=='m':
            C.extend([P*R,sat(A@p[10:15],kind[1])]);
        elif kind[0]=='n':
            C=[pol(X,2),sat(P,kind[1])];
            if kind.endswith('d'): C.extend([sat(R,kind[2])]);
        elif kind[0]=='h':
            C.extend([P*R, np.tanh(P),np.tanh(R),np.tanh(A@p[10:15])]);
        elif kind[0]=='f':
            T=P*R; W=A@p[10:15]; C.extend([T/(1+W*W)]);
        return np.column_stack(C)
    def res(p):
        t=des(aa,p);return t@ls(t,y)-y
    seed=np.random.default_rng(235); starts=[]
    for i in range(5):
        j1,j2=(a,b) if kind[0]=='q' else (u,v)
        first=np.r_[0,j1,0,j2]*[.3,.65,1.,1.4,2.][i]
        if kind[0] in 'mhf': first=np.r_[first,0,co[1:5]*([.15,.4,.7,1.,1.3][i])];
        if i: first+=seed.normal(0,.13,len(first))
        starts.append(first)
    best=None
    for start in starts:
        p=least_squares(res,start,bounds=(-4,4),max_nfev=150,ftol=3e-6).x
        g=des(aa,p); w=ls(g,y); loss=np.mean((g@w-y)**2)
        if best is None or loss<best[0]:best=loss,des(az,p)@w,p,w
    return best
if __name__=='__main__':
    for t in [0,1]:
        yy=Y[:,t]; rows=[]
        for kind in ['p'+a+b+c for a,b in product('itsr',repeat=2) if a+b!='ii' for c in ['','d']]+['qtt','qssd','qt​​td'.replace('\u200b',''),'qtrd','mt','ms','mr','nt','ns','nr','nttd','ht','ft']:
            q,w,p,co=readout(x[:52],x[52:],yy[:52],kind); rows.append((np.sqrt(q),np.mean(abs(w-yy[52:])),kind,p,co));
        for r in sorted(rows,key=lambda r:r[1])[:8]:print(t,'val sorted',r,flush=True)
        for r in sorted(rows,key=lambda r:r[0])[:6]:print(t,'fit sorted',r[:3],flush=True)
