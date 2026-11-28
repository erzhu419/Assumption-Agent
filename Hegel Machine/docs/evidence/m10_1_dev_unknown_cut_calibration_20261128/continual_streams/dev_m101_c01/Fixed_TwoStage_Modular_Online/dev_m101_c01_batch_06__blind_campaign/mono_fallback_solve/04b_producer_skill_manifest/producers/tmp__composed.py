from probe import *
from scipy.linalg import qr

def panel_c(s):
 p=s['panel_c'];u=p['u'];out=[]
 for key,op,init in [('v1',lambda x,k:x/(1+k*abs(x)),.83),('v2',lambda x,k:np.cos(x*k),1.25)]:
  def basis(x,k):return np.c_[design(x,1),op(x,k)]
  v=p[key]
  def err(k):
   f=basis(u,k);return (f@np.linalg.lstsq(f,v,rcond=None)[0]-v).ravel()
  fit=least_squares(err,np.full(4,init),bounds=(.15,3.1),gtol=1e-12);k=fit.x;w=np.linalg.lstsq(basis(u,k),v,rcond=None)[0];print(key,'rms',np.std(fit.fun),'k',k,'coefficients',w.T);out.append(lambda x,k=k,w=w,op=op:np.c_[design(x,1),op(x,k)]@w)
 return out

def spaces(s,x):
 a=s['panel_a'];A,_=robust(design(a['u'],1),a['v']);A[1:]=np.diag(A[1:].diagonal()); b=s['panel_b'];B=np.linalg.lstsq(design(b['u'],1),b['v'],rcond=None)[0]
 af=lambda X,M:X@M[1:]+M[0];ai=lambda X,M:(X-M[0])@np.linalg.inv(M[1:]);
 return {'x':x,'a':af(x,A),'ai':ai(x,A),'b':af(x,B),'bi':ai(x,B),'ba':af(af(x,A),B),'bai':af(ai(x,A),B),'bia':ai(af(x,A),B),'biai':ai(ai(x,A),B),'ab':af(af(x,B),A),'aib':ai(af(x,B),A),'abi':af(ai(x,B),A),'aibi':ai(ai(x,B),A)}

def folds(x):
 yield np.arange(56),np.arange(56,72)
 rng=np.random.default_rng(102)
 for j in range(4):
  i=rng.permutation(len(x));yield i[:56],i[56:]
 yield np.argsort((x*x).sum(1))[:56],np.argsort((x*x).sum(1))[56:]

def score(F,Y,ss):
 all=[]
 for ridge in (1e-10,1e-5,1e-3,1e-2):
  err=[]
  for i,j in ss:
   scale=np.maximum(.03,F[i].std(0));scale[0]=1;Q=F/scale
   W=np.linalg.solve(Q[i].T@Q[i]+ridge*np.eye(Q.shape[1]),Q[i].T@Y[i]);err.append(abs(Q[j]@W-Y[j]).mean(0))
  all.append((ridge,np.array(err)))
 return all

if __name__=='__main__':
 x,y,z=[S['train'][k] for k in ('x','y','traj')];c1,c2=panel_c(S);ss=list(folds(x));Y=np.c_[y,z];D=design(x,2);results=[]
 for key,q in spaces(S,x).items():
  g,h=c1(q),c2(q); H=np.c_[g,h];
  for name,F in {'lin':np.c_[np.ones(len(x)),H], 'linx':np.c_[design(x,1),H], 'g2':design(g,2), 'h2':design(h,2), 'gh2':design(H,2), 'base+':np.c_[D,H], 'base+g^2':np.c_[D,H,poly(g,2)],'base+h^2':np.c_[D,H,poly(h,2)], 'xg':np.c_[D,H,(q[:,:,None]*g[:,None,:]).reshape(-1,16)], 'g3': design(g,3)}.items():
   for r,e in score(F,Y,ss):
    v=np.c_[e[:,0],e[:,1:].mean(1)];results.append((key,name,r,v))
 for j in range(2):
  print('TARGET',j)
  for k,n,r,v in sorted(results,key=lambda a:.45*a[-1][0,j]+.3*a[-1][:-1,j].mean()+.25*a[-1][:-1,j].max())[:17]:print(k,n,r,np.round(v.T,5))
