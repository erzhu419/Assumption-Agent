exec(open('/tmp/probe08.py').read().split('for f in sorted')[0]); m=load(glob.glob('/tmp/hegel_continual_archive/*06_solver.py')[0]); q=m._invert(x,m._fit_calibration(s['panel_a'])); b=m._panel_b(s['panel_b'],q); c=b*np.roll(q,-1,axis=1); C=c.sum(1); base=np.c_[poly(b,1),C]; full=np.c_[poly(b,1),c]
folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(52),i),i) for i in np.array_split(np.random.default_rng(192).permutation(52),4)]
def cv(f,Y=y):
 return np.array([err(f[j]@ls(f[i],Y[i])-Y[j]).mean() for i,j in folds]);
def fmt(e):return np.round(e,5)
print('C range',C.min(),C.max(),'co',ls(full[:52],y[:52]),'simple co',ls(base[:52],y[:52]));print('base',fmt(cv(base)),'full',fmt(cv(full)))
extras={'C':C,'Qnorm':np.sum(q*q,1),'Bnorm':np.sum(b*b,1),'Qmean':q.mean(1),'Bmean':b.mean(1),'BCnorm':np.sum(c*c,1),'Cabs':abs(c).sum(1),'state':full@ls(full,z[:,-1])}
for j in range(4):extras.update({f'q{j}':q[:,j],f'b{j}':b[:,j],f'c{j}':c[:,j]})
for k in [2,3,4,5,6]:print('C poly',k,fmt(cv(np.c_[base,*[C**i for i in range(2,k+1)]])))
results=[]
for name,a in extras.items():
 for g,v in [('sq',a*a),('cu',a**3),('tanh',np.tanh(a)),('sin',np.sin(a)),('abs',abs(a)),('rat',a/(1+a*a)),('sig',1/(1+a*a))]:
  for tag,f in [('add',np.c_[base,v]),('mulC',np.c_[base,C*v]),('mulB',np.c_[base,b*v[:,None]]),('mulAll',np.c_[base,base*v[:,None]])]:
   e=cv(f);results.append((e[1:].mean(),name,g,tag,e))
print('screen: ranked by folds of first 52 ONLY')
for a,k,g,tag,e in sorted(results,key=lambda x:x[0])[:36]:print(k,g,tag,fmt(e))
print('op coeffs by time for structural model:');print(ls(full,z))
print('sorted residual vs C:'); print(np.c_[C,y-base@ls(base,y)][np.argsort(C)])
