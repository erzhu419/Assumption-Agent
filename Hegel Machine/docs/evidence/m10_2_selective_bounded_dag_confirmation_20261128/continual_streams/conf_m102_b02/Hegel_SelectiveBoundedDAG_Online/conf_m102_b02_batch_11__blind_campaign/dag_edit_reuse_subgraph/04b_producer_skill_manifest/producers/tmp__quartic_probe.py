exec(open('/tmp/panel_probe.py').read().split('for k,v in dict')[0])
from sklearn.linear_model import OrthogonalMatchingPursuit, Ridge
from sklearn.model_selection import KFold
for name,q in [('A',x),('B',z),('B0',x@b[1:]),('AB',np.c_[x,z]),('AB0',np.c_[x,x@b[1:]])]:
 tt=[c for j in range(1,5) for c in cwr(range(q.shape[1]),j)]
 D=np.column_stack([q[:,c].prod(1) for c in tt]);
 print('\n',name,flush=True)
 for n in [5,10,16,22,30,38]:
  scores=[]
  for tr,te in [(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=13).split(x)]:
   X=D[tr]-D[tr].mean(0); sc=np.maximum(np.linalg.norm(X,axis=0),1e-10); X=X/sc; Y=y[tr]-y[tr].mean(0); ys=np.linalg.norm(Y,axis=0); Y=Y/ys
   sel=[]; R=Y.copy()
   for _ in range(n):
    cor=np.sum((X.T@R)**2,axis=1); cor[sel]=-1; sel.append(cor.argmax()); a=np.linalg.lstsq(X[:,sel],Y,rcond=None)[0]; R=Y-X[:,sel]@a
   e=(D[te][:,sel]-D[tr][:,sel].mean(0))/sc[sel]@a*ys+y[tr].mean(0)-y[te]; scores.append([abs(e[:,0]).mean(),abs(e[:,1:]).mean()]);
   if n==16 and len(scores)==1: print('terms:',[tt[j] for j in sel]); print('scalar coefficients:',np.round(a[:,0]*ys[0]/sc[sel],3));
  print(n,np.round(scores[0],4),np.round(np.max(scores,axis=0),4),flush=True)
