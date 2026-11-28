from explore14 import *
a=s['panel_a'];u,v=a['u'],a['v'];D=aff(u); rows=np.arange(len(u))
for _ in range(10):
 e=np.linalg.norm(D@ls(D[rows],v[rows])-v,axis=1);rows=np.flatnonzero(e<2)
print('A excluded',np.setdiff1d(np.arange(len(u)),rows));print('A rms',np.std((D@ls(D[rows],v[rows])-v)[rows],axis=0))
for j in range(4):
 x=u[rows,j];w=v[rows,j]; best=[]
 for k in range(4):
  z=u[rows,k]
  fs={'lin':z,'sin24':np.sin(2.4*z),'sin215':np.sin(2.15*z),'sin26':np.sin(2.6*z),'tanh18':np.tanh(1.8*z),'square':z*z,'cube':z**3,'product':z*x,'cos26':np.cos(2.6*z)}
  gs={'lin':x,'tanh18':np.c_[x,np.tanh(1.8*x)],'cube':np.c_[x,x**3],'square':np.c_[x,x*x],'sin24':np.c_[x,np.sin(2.4*x)],'all':np.c_[x,x*x,x**3,x**5]}
  for f,b in fs.items():
   for g,c in gs.items():
    B=aff(np.c_[c,b]);co=ls(B,w);best.append((np.std(B@co-w),k,f,g,co))
 print(j,*sorted(best,key=lambda t:t[0])[:5],sep='\n')
for name,key in [('panel_c','v1'),('panel_c','v2')]:
 d=s[name]; u,v=d['u'],d[key];r=[]
 f2={**funcs,'crossmix':(lambda x,p:np.tanh(p[0]*x)+p[1]*np.sin(p[2]*np.roll(x,1,axis=1)),[1.75,.30,2.10]),'lin+asinh':(lambda x,p:np.c_[x,np.arcsinh(p[0]*x)],[2.15]),'rational+sin':(lambda x,p:np.c_[x/(1+p[0]*x*x),np.sin(p[1]*x)],[.22,2.15])}
 for k,(fn,p) in f2.items():r.append((*score(u,v,fn,p)[:3],k))
 print(name,key,*sorted(r,key=lambda q:q[1])[:10],sep='\n')
