exec(open('/tmp/cuts.py').read().split('res=[]')[0])
p,r=D['pc'],D['rc']; a=aff(np.c_[p,r]); cross=(p[:,:,None]*r[:,None,:]).reshape(-1,16); dot=cross[:,::5].sum(1); F=np.c_[a,dot];
for tag,f in [('dot',F),('diag',np.c_[a,cross[:,::5]]),('full',np.c_[a,cross])]:
 w=ls(f[I],Y[I]); print(tag,'weights',np.round(w,4)); print('mae',np.mean(abs((f[J]@w-Y[J])[:,1:]@Q.T)))
print('Top single term searches from dot basis');
for j in range(16):
 f=np.c_[F,cross[:,j]]; w=ls(f[I],Y[I]); e=f[J]@w-Y[J]; print(divmod(j,4),np.mean(abs(e[:,1:]@Q.T)),np.std(f[I]@w-Y[I],axis=0))
f=np.c_[F,cross[:,1]]
for j in range(16):
 f2=np.c_[f,cross[:,j]]; w=ls(f2[I],Y[I]); e=f2[J]@w-Y[J]; print('second',divmod(j,4),np.mean(abs(e[:,1:]@Q.T)),np.std(f2[I]@w-Y[I],axis=0))
