exec(open('/tmp/graph05.py').read().split('for xn,U')[0])
p,q=C(z)@c,D(z)@d
dot=np.sum(p*q,axis=1); cross=(p[:,:,None]*q[:,None,:]).reshape(len(p),16)
F=one(np.c_[p,q,dot]); G=one(np.c_[p,q,cross]); print('scalar',ls(F[:52],Y[:52,0]));print('amplitudes',np.round(ls(G[:52],A[:52]),4));
blocks=[('p',one(p)),('q',one(q)),('p_dot',one(np.c_[p,dot])),('q_dot',one(np.c_[q,dot])),('both_dot',F),('all',G)]
for k in range(16): blocks.append((f'q_{k//4}{k%4}',one(np.c_[q,cross[:,k]])))
def fit(F,G,idx,out):
 def mat(r,j):
  h=H(r); return np.concatenate([F[j,None,:]*h[None,:,0,None],G[j,None,:]*h[None,:,1,None]],axis=2).reshape(len(j)*len(t),-1)
 obs=T[idx].ravel()
 def res(r):
  m=mat(r,idx);return m@ls(m,obs)-obs
 r=least_squares(res,[1.81,.42],bounds=([.8,.05],[4,.78]),ftol=1e-10).x
 m=mat(r,idx); b=ls(m,obs); e=abs((mat(r,out)@b).reshape(len(out),-1)-T[out]);return np.mean(res(r)**2)**.5,e.mean(),e.mean(0).max(),r,b
idx=np.arange(52);out=np.arange(52,72); rows=[]
for a,f in blocks[:6]:
 for b,g in blocks:
  v=fit(f,g,idx,out); rows.append((v[0],v[1],v[2],a,b,v[3],v[4]))
for v in sorted(rows,key=lambda x:x[0])[:12]: print(v[:6],np.round(v[6],5))
