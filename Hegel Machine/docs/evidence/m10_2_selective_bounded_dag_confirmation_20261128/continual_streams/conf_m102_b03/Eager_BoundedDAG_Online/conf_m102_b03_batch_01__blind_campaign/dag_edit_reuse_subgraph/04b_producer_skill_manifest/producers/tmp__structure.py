import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task'); import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=m._ls; ones=m._ones
tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; n=len(x)
a,b=m._align(s['panel_a']); B=ls(ones(s['panel_b']['u']),s['panel_b']['v']); z=(x-a)/b
p=s['panel_c']; u=p['u']; i,j=np.triu_indices(4); Q=lambda z:np.c_[ones(z),z[:,i]*z[:,j]]
coef=ls(Q(u),p['v1']); print('quadratic coefficients',np.round(coef,5)); print('trajectory SV',np.linalg.svd(t,compute_uv=False));
coords={'raw':x,'Ainv':z,'B':ones(z)@B,'Binv':(z-B[0])@np.linalg.inv(B[1:]),'Braw':ones(x)@B,'Binvraw':(x-B[0])@np.linalg.inv(B[1:])}
cf={k:Q(v)@coef for k,v in coords.items()}; df={k:m._branch(p,v,2) for k,v in coords.items()}; cuts=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(np.random.default_rng(92).permutation(n),4)]
def score(f,target):
 pred=[f[v]@ls(f[r],target[r])-target[v] for r,v in cuts]
 return tuple(np.mean(abs(q)) for q in pred)
Y=[]; T=[]
for ckey,dkey in itertools.product(coords,repeat=2):
 c,d=cf[ckey],df[dkey]; cross=(d[:,:,None]*c[:,None,:]).reshape(n,-1)
 f=ones(np.c_[d,c]); dot=(c*d).sum(1); diag=c*d
 for key,F in [('linear',f),('dot',np.c_[f,dot]),('diag',np.c_[f,diag]),('cross',np.c_[f,cross]),('allquad',np.c_[f,cross,c[:,i]*c[:,j],d[:,i]*d[:,j]])]:
  Y.append((score(F,y),ckey,dkey,key)); T.append((score(F,t),ckey,dkey,key))
for name,res in [('Y',Y),('T',T)]:
 print(name,'best by worst fold');
 for sc,*info in sorted(res,key=lambda q:max(q[0]))[:12]: print(info,np.round(sc,6))
