from search import *
from itertools import combinations
from scipy.linalg import qr
from sklearn.linear_model import Lasso, ARDRegression
u,v=s['panel_a']['u'],s['panel_a']['v']; w=np.roll(u,-1,axis=1); ww=np.roll(x,-1,axis=1)
a=np.column_stack([np.c_[np.ones(len(x)),x[:,j],ww[:,j],x[:,j]*ww[:,j],x[:,j]**3]@fit(np.c_[np.ones(len(keep)),u[keep,j],w[keep,j],u[keep,j]*w[keep,j],u[keep,j]**3],v[keep,j]) for j in range(4)])
zsets={'x':x,'a':a,'b':B(x),'ba':B(a),**{k:v for k,v in zdict.items() if k not in ('x','a')}}
def bank(zsets,grid=(.45,.85,1.3,1.8,2.4,3.2)):
 fs=[]; names=[]
 def add(v,n):fs.append(v);names.append(n)
 for key,z in zsets.items():
  for d in (1,2,3):
   for ix in cr(range(4),d):add(np.prod(z[:,ix],axis=1),(key,'poly',ix))
  dirs=[((j,),z[:,j]) for j in range(4)]
  dirs += [((j,k,c),z[:,j]+c*z[:,k]) for j,k in combinations(range(4),2) for c in (-1,-.5,.5,1)]
  for ix,w in dirs:
   for g in grid:
    h=w*g
    for fn,v in [('sin',np.sin(h)),('cos',np.cos(h)),('tanh',np.tanh(h)),('rat',1/(1+h*h)),('orat',w/(1+h*h))]:
     add(v,(key,fn,ix,g))
     if len(ix)==1 and key in ('x','a','b','ba','c','ca'):
      for j in range(4):
       add(z[:,j]*v,(key,fn,ix,g,j))
  if key in ('x','a','b','ba'):
   for j,k in product(range(4),repeat=2):
    for g in (.3,.7,1.3,2.2):
     add(z[:,j]/(1+g*z[:,k]**2),(key,'ratio',j,k,g));add(z[:,j]**2/(1+g*z[:,k]**2),(key,'ratio2',j,k,g))
 return np.column_stack(fs),names
F,names=bank(zsets)
def prep(i,q):
 h=np.linalg.qr(q[i])[0]; co=fit(q[i],F[i]); rot=F-q@co; scale=np.linalg.norm(rot[i],axis=0);return rot/np.where(scale>1e-6,scale,np.inf)
def trail(i,j,q,Y,count=28,multi=True):
 g=prep(i,q); G=g[i]; beta=fit(q[i],Y[i]); R=Y[i]-q[i]@beta; D=R.copy(); scale=np.sqrt(np.mean(R**2,axis=0)); active=[]; out=[]
 for k in range(count):
  s=np.sum((G.T@(D/scale))**2,axis=1);s[active]=-1;active.append(int(np.argmax(s))); W=fit(G[:,active],R);D=R-G[:,active]@W;out.append(q[j]@beta+g[j][:,active]@W)
 return np.array(out),active
if __name__=='__main__':
 traces=[]
 for qname,q in [('lin',poly(x,1)),('quad',poly(x,2)),('quad+c',np.c_[poly(x,2),zsets['c']])]:
  for joint in (True,False):
   E=[]
   for k,(i,j) in enumerate(fds):
    b=bases[tuple(i)];Y=np.c_[y,tr@b];
    if joint:p,ac=trail(i,j,q,Y)
    else:
     pp,ac=zip(*(trail(i,j,q,Y[:,m:m+1]) for m in range(3)));p=np.concatenate(pp,axis=2)
    if k==0:print(qname,joint,'first terms',[[names[m] for m in row[:7]] for row in (ac if not joint else [ac])])
    E.append(np.c_[np.mean(abs(p[:,:,0]-y[j]),axis=1),np.mean(abs(p[:,:,1:]@b.T-tr[j]),axis=(1,2))])
   E=np.array(E); print(qname,joint,[(m+1,E[0,m].round(4).tolist(),E[:,m].mean(0).round(4).tolist(),round(E[:,m].max(),4)) for m in (2,4,6,8,11,15,21,27)])
