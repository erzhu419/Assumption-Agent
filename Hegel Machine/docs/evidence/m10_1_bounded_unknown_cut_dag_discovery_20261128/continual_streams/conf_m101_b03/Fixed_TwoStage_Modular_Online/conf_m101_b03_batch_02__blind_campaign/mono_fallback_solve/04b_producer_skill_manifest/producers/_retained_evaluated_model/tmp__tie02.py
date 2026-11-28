exec(open('/tmp/check02.py').read().split("if __name__")[0])
base=np.c_[m._affine(z),m._cyclic(z),z*z]; pq=(p[:,:,None]*q[:,None,:]).reshape(72,16); dot=np.sum(p*q,axis=1); F=np.c_[base,q]; n=len(base); traj=tr['traj']; y=tr['y']; t=s['eval_times']; rng=np.random.default_rng(202)
splits=[all_[52:],*np.array_split(all_,4),*np.array_split(rng.permutation(all_),5),np.argsort(np.linalg.norm(z,axis=1))[-20:],np.argsort(np.max(abs(z),axis=1))[-20:]]
models={}
for hold in splits:
 a=np.setdiff1d(all_,hold); h,ab=m._temporal(t,traj[a]); r=m._fit(np.c_[base[a],ab],y[a]);
 for joint in [False,True]:
  if not joint:
   abj=ab.copy(); abj[:,1]=.5*(ab[:,1]+(y[a]-np.c_[base[a],ab[:,0]]@r[:-1])/r[-1]);
  for typ,fast in [('full',np.c_[F,pq]),('diag',np.c_[F,p*q]),('dot',np.c_[F,dot]),('dot_simple',np.c_[m._affine(z),p,q,dot])]:
   for sl,slow in [('full',np.c_[F,pq]),('small',np.c_[F,pq[:,1]]),('quadq',F),('state',np.c_[m._affine(z),q,pq[:,1]])]:
    if typ=='full' and sl!='full':continue
    gs,gf=slow[a],fast[a]; d1,d2=gs.shape[1],gf.shape[1];
    if joint:
     A=np.concatenate([gs[:,None,:]*h[None,:,0,None],gf[:,None,:]*h[None,:,1,None]],axis=-1).reshape(-1,d1+d2); B=np.c_[r[-2]*gs,r[-1]*gf]; X=np.r_[A,B]; Y=np.r_[traj[a].ravel(),y[a]-base[a]@r[:-2]]; w=m._fit(np.r_[X,np.diag(.005*np.linalg.norm(X,axis=0))],np.r_[Y,np.zeros(X.shape[1])]); abp=np.c_[slow@w[:d1],fast@w[d1:]]
    else:
     def reg(g,y):
      G=g[a]; return m._fit(np.r_[G,np.diag(.01*np.linalg.norm(G,axis=0))],np.r_[y,np.zeros(G.shape[1])])
     abp=np.c_[slow@reg(slow,abj[:,0]),fast@reg(fast,abj[:,1])]
    yp=base@r[:-2]+abp@r[-2:]; yp0=np.mean(abs(yp[hold]-y[hold])); st=abs(abp[hold]@h.T-traj[hold]); models.setdefault((typ,sl,joint),[]).append([yp0,st.mean(),st.mean(0).max()]);
for k,v in sorted(models.items(),key=lambda p:np.max(np.asarray(p[1]),axis=0)[1]):
 v=np.array(v); print(k,'final20',v[0].round(5),'worst',v.max(0).round(5),'mean',v.mean(0).round(5),'extreme',v[-2].round(5))
