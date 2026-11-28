import numpy as np,pickle,itertools
N=pickle.load(open('/tmp/node03_nested.pkl','rb')); L=pickle.load(open('/tmp/node03_loo.pkl','rb')); A=pickle.load(open('/tmp/node03_folds.pkl','rb'))
rows=[r for _,r in N]+[r for _,r in L[:4]]+[(a['poly'],b['smooth'],v,y,t) for a,b,y,t,v in A[1:]]
def features(p,b,v):
 s=b[:,0]; d=p[:,0]-s; t=(.4*p[:,1:]+.6*b[:,1:])@v; e=(p[:,1:]-b[:,1:])@v
 a=t[:,-1]; k=t[:,0]-.2*a; f=e[:,-1]; g=e[:,0]-.2*f
 q=lambda x,c: np.clip(x/c,-1,1)
 S=q(s,1); D=q(d,.05); F=q(f,.025); G=q(g,.008); K=q(k,.025); B=q(a,1)
 return dict(dis=D,absdis=abs(D),rawdis=np.clip(d,-.2,.2)/.05,offset=np.ones_like(s),mag=abs(S),squared=S*S,odd=S,curv=S**3,positive=np.maximum(S,0),negative=np.minimum(S,0),level=B,shape=K,endgap=F,shapegap=G,endmag=abs(F),shapemag=abs(G),dis_odd=D*S,dis_mag=D*abs(S),dis_level=D*B,dis_shape=D*K,dis_endgap=D*F,dis_shapegap=D*G,dis_conc=D*abs(F),endgap_odd=F*S,shapegap_odd=G*S,endgap_level=F*B,shapegap_level=G*B,level_mag=abs(B),level_odd=B*S,shape_odd=K*S,level_shape=B*K)
parts=[(features(p,b,v),y-b[:,0],p[:,0]-b[:,0],len(y)) for p,b,v,y,t in rows]; names=list(parts[0][0]); X=np.column_stack([np.concatenate([f[k] for f,_,_,_ in parts]) for k in names]); Y=np.concatenate([r for _,r,_,_ in parts]); D=np.concatenate([d for _,_,d,_ in parts]); ns=np.array([n for *_,n in parts]); ix=np.r_[0,np.cumsum(ns)[:-1]]; dev=np.arange(len(ns))!=4
res=[]
for w in [-.15,0,.15,.3,.45,.6]:
 R=Y-w*D
 for i,j in itertools.combinations(range(len(names)),2):
  a,b=np.meshgrid(np.arange(-4,5)*.0125,np.arange(-4,5)*.0125); a=a.ravel(); b=b.ravel()
  err=np.add.reduceat(abs(R[:,None]-X[:,i,None]*a-X[:,j,None]*b),ix,axis=0)/ns[:,None]
  for m in np.flatnonzero(err[4]<=.03785):
   z=err[:,m]; rank=np.mean(z[dev])+.25*np.max(z[dev])+.015*(abs(a[m])+abs(b[m])); res.append((rank,z[4],w,names[i],float(a[m]),names[j],float(b[m]),z))
print('passing',len(res)); res.sort(key=lambda x:x[0]);
for r in res[:16]: print(*r[:-1],np.round(r[-1],5));
if res: pickle.dump(res[:100],open('/tmp/node03_contrast_candidates.pkl','wb'))
