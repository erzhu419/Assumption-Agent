exec(open('/tmp/grammar6.py').read().split('for pk,names')[0]);
np.set_printoptions(precision=4,suppress=True)
labels=np.r_[np.arange(len(x))]; folds=[labels[52:]]+list(np.array_split(np.random.default_rng(17).permutation(labels),6))+[j for k in range(4) for j in (np.argsort(x[:,k])[:12],np.argsort(x[:,k])[-12:])];
def cve(A):
 es=[]
 for j in folds:
  i=np.setdiff1d(labels,j); w=np.linalg.lstsq(A[i],Y[i],rcond=None)[0]; e=abs(A[j]@w-Y[j]);es.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(axis=0).max()]);
 return np.array(es)
for powers in [(1,3,5),(1,2,3,4,5),(1,3,5,7),(1,2,3,4,5,6,7)]:
 for ix in [[],pairs,pairs+triples]:
  A=np.c_[one(x),*[x**k for k in powers],*[x[:,j].prod(1) for j in ix]]; e=cve(A); print(powers,len(ix),e[0],e[:7].max(0),e[7:].max(0));
A=features['+diag5'](x); W=np.linalg.lstsq(A,Y,rcond=None)[0]; print('coef y, first, final:');print(W[:,[0,1,-1]]);
for key in ('panel_a','panel_b','panel_c'):print(key,{k:np.shape(v) for k,v in s[key].items()})
