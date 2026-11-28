from operators import *
from itertools import product
c=s['panel_c'];u=c['u']; V=c['v2']; Q=lambda u:F(np.c_[u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]])
C1=fit(Q(u),c['v1']); print('C1 coeff',C1.T,'validation',report(Q(u),c['v1'],18));
models=[]
for nam,fn in {'cs':np.cos,'gauss':lambda z:np.exp(-z*z),'rat':lambda z:1/(1+z*z),'rat4':lambda z:1/(1+z**4)}.items():
 for base in [False,True]:
  for mode in range(3):
   def D(z,p):
    w=fn(z*p[0 if mode<2 else slice(0,4)]+(0 if mode==0 else p[1:] if mode==1 else p[4:])); return F(np.c_[z,w]) if base else F(w)
   res=[]
   for init in [0.8,1.25,2.5,3.7]:
    p=[init] if mode==0 else [init]+[0]*4 if mode==1 else [init]*4+[0]*4
    def err(p):
     H=D(u[:19],p);return (H@fit(H,V[:19])-V[:19]).ravel()
    q=least_squares(err,p,max_nfev=100); e=report(D(u,q.x),V,19).mean(); res.append((e,q.x))
   e,p=min(res,key=lambda v:v[0]); print('C2',nam,base,mode,e,p);models.append((e,nam,base,mode,p,fn))
e,nam,base,mode,p,fn=min(models,key=lambda v:v[0]); print('winner',e,nam,base,mode,p)
def H(z):
 w=fn(z*p[0 if mode<2 else slice(0,4)]+(0 if mode==0 else p[1:] if mode==1 else p[4:]));return F(np.c_[z,w]) if base else F(w)
C2=fit(H(u),V); print('C2 coeff',C2.T)
def f1(x):return Q(x)@C1
def f2(x):return H(x)@C2
if __name__=='__main__':
 x=s['train']['x']; z=(x-a)/b; Y=np.c_[s['train']['y'],s['train']['traj']]; R=[]
 def add(D,name):
  scale=np.maximum(np.std(D[:52],axis=0),.001); D=D/scale
  for l in [0,.0001,.01]:
   if l==0:
    if D.shape[1]>45: continue
    pred=D[52:]@fit(D[:52],Y[:52]); tr=D[:52]@fit(D[:52],Y[:52]);
   else:
    K=D@D.T; coeff=np.linalg.solve(K[:52,:52]+l*np.eye(52),Y[:52]); pred=K[52:,:52]@coeff; tr=K[:52,:52]@coeff
   E=abs(pred-Y[52:]); E0=abs(tr-Y[:52]);R.append((E[:,0].mean(),E[:,1:].mean(),E0[:,0].mean(),E0[:,1:].mean(),name,l))
 Z={'z':z,'Bz':Bf(z),'x':x,'Binvz':Binv(z),'Blinz':z@B[1:]}
 for name,w in Z.items():
  v,g=f1(w),f2(w)
  for k,j in [('v',v),('g',g),('vg',np.c_[v,g]),('zvg',np.c_[w,v,g]),('vz',np.c_[w,v]),('gz',np.c_[w,g])]:
   add(F(j),(name,k,'linear')); add(Q(j) if j.shape[1]==4 else F(np.c_[j,*[j[:,i]*j[:,k] for i in range(j.shape[1]) for k in range(i,j.shape[1])]]),(name,k,'quad'))
  X=[w,v,g]; L=['w','v','g'];
  for i,j in product(range(3),repeat=2):
   d=X[i];e=X[j]; m=(d[:,:,None]*e[:,None,:]).reshape(len(w),-1); 
   add(F(np.c_[w,v,g,m]),(name,L[i]+L[j],'outer'))
   add(F(np.c_[w,v,g,np.sum(d*e,axis=1),d*e]),(name,L[i]+L[j],'diag'))
   for k in [.1,.22,.5,1,1.25,2]:
    Hs=[d/(1+k*e**2), d/(1+k*np.sum(e**2,axis=1,keepdims=True))]
    for ind,block in enumerate(Hs):add(F(np.c_[w,v,g,block]),(name,L[i]+L[j],k,ind))
 for k in (0,1):print('best',k,*sorted(R,key=lambda row: row[k])[:15],sep='\n')
