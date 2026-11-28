exec(open('/tmp/grammar.py').read().split('rows=[]')[0])
from itertools import combinations_with_replacement
N=52
funcs={'rat':lambda z,c:z/(1+c*z*z),'sat':lambda z,c:np.tanh(c*z)/c,'soft':lambda z,c:z/(1+c*abs(z)),'gauss':lambda z,c:z*np.exp(-c*z*z),'root':lambda z,c:z/np.sqrt(1+c*z*z),'sin':lambda z,c:np.sin(c*z)/c}
rows=[]
for key in ['a','a1','x1','ba','ba1']:
 z=trans[key][:72];z=z/z[:N].std(0)
 for fam,fn in funcs.items():
  for deg,mode in [(2,0),(2,1),(3,0),(2,2)]:
   for cols in [(0,1,2),(0,),(1,),(2,)]:
    yt=Y[:N,cols]; scale=yt.std(0)
    def F(q):
     w=fn(z+q[4:],q[:4]); m=[np.prod(w[:,ids],axis=1) for k in range(deg+1) for ids in combinations_with_replacement(range(4),k)];
     if mode==1:m.extend([z,z*z]);
     if mode==2:m.extend([z,z*z,np.prod(z[:,[0,3]],axis=1),np.prod(z[:,[1,2]],axis=1)]);
     return np.column_stack(m)
    def obj(q):
     X=F(q)[:N];return ((X@fit(X,yt)-yt)/scale).ravel()
    best=None
    for start in [.12,.65,1.3]:
     rr=least_squares(obj,np.r_[np.full(4,start),np.zeros(4)],bounds=(np.r_[np.full(4,.005),np.full(4,-1.4)],np.r_[np.full(4,2.8),np.full(4,1.4)]),max_nfev=45)
     if best is None or np.linalg.norm(rr.fun)<best[0]:best=np.linalg.norm(rr.fun),rr.x
    _,q=best; ff=F(q); X=ff[:N]; P=np.linalg.pinv(X); coef=P@yt; h=np.sum(X*P.T,1); ei=np.mean(abs(X@coef-yt)/np.maximum(1-h,.05)[:,None],0)*N/(N-8); et=np.mean(abs(ff[N:]@coef-Y[N:,cols]),0)
    for k,j in enumerate(cols):rows.append((j,ei[k],et[k],key,fam,deg,mode,cols,q))
for j in range(3):
 R=[r for r in rows if r[0]==j];print('best',j);print(*sorted(R,key=lambda r:r[1])[:4],sep='\n');print('tail diagnostic');print(*sorted(R,key=lambda r:r[2])[:4],sep='\n')
