exec(open('/tmp/forward12.py').read().split('for name,x in')[0])
dot=np.sum(P*R,axis=1); F=D(P,dot); S=D(R); C=np.column_stack([P[:,i]*R[:,j] for i in range(4) for j in range(4)])
for fn,f in [('P+dot',F),('PR+dot',D(P,R,dot)),('PR+all',D(P,R,C))]:
 pred=f@ls(f[:n],Y[:n]); print(fn,'scalar train/test',np.mean(abs(pred[:n,0]-Y[:n,0])),np.mean(abs(pred[n:,0]-Y[n:,0])), 'w',ls(f[:n],Y[:n,0]))
def profile(k,f,q,m):
 h=H(k); d=np.concatenate([f[:m,None,:]*h[None,:,0,None],q[:m,None,:]*h[None,:,1,None]],axis=2).reshape(m*len(tt),-1)
 w=ls(d,A[:m].ravel()); return d@w-A[:m].ravel(),w
results=[]
for fn,f in [('P+dot',F),('PR+dot',D(P,R,dot))]:
 for i,q in [(-1,S)]+[(i,D(R,C[:,i])) for i in range(16)]+[(16,D(R,dot)),(17,D(R,C))]:
  o=least_squares(lambda k:profile(k,f,q,n)[0],opt.x,bounds=([0.8,.03],[5,1.0])); e,w=profile(o.x,f,q,n); pred=np.c_[f[n:]@w[:f.shape[1]],q[n:]@w[f.shape[1]:]]@H(o.x).T
  results.append((np.mean(abs(e)),np.mean(abs(pred-A[n:])),fn,i,o.x,w))
for row in sorted(results,key=lambda a:a[0])[:12]:print('trajectory',row)
