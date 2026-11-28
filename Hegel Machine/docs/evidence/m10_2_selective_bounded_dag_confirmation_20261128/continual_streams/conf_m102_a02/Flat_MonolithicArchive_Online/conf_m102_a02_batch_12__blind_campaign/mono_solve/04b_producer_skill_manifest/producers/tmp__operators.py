exec(open('/tmp/explore.py').read().split("print('rates'")[0])
p=s['panel_a']; u,v=p['u'],p['v']; j,k=np.triu_indices(len(u),1); m=np.median((v[j]-v[k])/(u[j]-u[k]),0); b=np.median(v-u*m,0); keep=np.max(abs(v-u*m-b),axis=1)<.01
m=np.sum((u[keep]-u[keep].mean(0))*(v[keep]-v[keep].mean(0)),0)/np.sum((u[keep]-u[keep].mean(0))**2,0); b=np.mean(v[keep]-u[keep]*m,0); z=(x-b)/m
families={
 'tanh':(lambda a,p:np.tanh(p[0]*a),[1.5],(.02,12)),
 'sin':(lambda a,p:np.sin(p[0]*a),[2.],(.02,12)),
 'cos':(lambda a,p:np.cos(p[0]*a),[2.],(.02,12)),
 'rat':(lambda a,p:a/(1+p[0]*abs(a)**p[1]),[1.,2.],([.01,.25],[12,8])),
 'exp':(lambda a,p:a*np.exp(-p[0]*abs(a)**p[1]),[1.,2.],([.01,.25],[12,8])),
 'soft':(lambda a,p:a/np.sqrt(1+p[0]*a*a),[1.],(.01,20)),
 'pow':(lambda a,p:np.sign(a)*abs(a)**p[0],[2.],(.1,8)),
 'gauss':(lambda a,p:np.exp(-p[0]*a*a),[1.],(.01,20)),
}
for name,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[name]['u'],s[name][key]; print(name,key,'v sd',v.std(), 'vsv',np.linalg.svd(v,compute_uv=False)); out=[]
 for d in [1,2,3,4,5]:
  f=np.c_[np.ones(len(u)),*[u**i for i in range(1,d+1)]]; print('separable poly',d,np.sqrt(np.mean((f@ls(f,v)-v)**2)))
 for ext in [0,1,2]:
  for nm,(op,p0,bd) in families.items():
   def feat(u,p):return np.c_[np.ones(len(u)),op(u,p)] if ext==0 else np.c_[aff(u) if ext==1 else poly(u,2),op(u,p)]
   def obj(p):
    d=feat(u,p);return (d@ls(d,v)-v).ravel()
   f=least_squares(obj,p0,bounds=bd,max_nfev=100); d=feat(u,f.x); pred=feat(z,f.x)@ls(d,v)
   out.append((np.std(f.fun),nm,ext,f.x.round(4),score(aff(pred),T),score(poly(pred,2),T),score(poly(pred,2),y)))
 for row in sorted(out,key=lambda q:q[0])[:13]:print(row)
print('z range',z.min(0),z.max(0),'y sorted',np.sort(y).round(3));
