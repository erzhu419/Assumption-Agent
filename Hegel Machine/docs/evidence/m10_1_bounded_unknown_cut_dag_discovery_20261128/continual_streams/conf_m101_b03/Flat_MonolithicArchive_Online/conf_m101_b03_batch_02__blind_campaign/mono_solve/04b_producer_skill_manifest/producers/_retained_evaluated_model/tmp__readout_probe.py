from probe import *
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from itertools import combinations
np.set_printoptions(precision=4,suppress=True)
z,b,_,w=coords(T['x']); pc=S['panel_c']; uc=pc['u']; print('C1 rows', ['1']+[''.join(map(str,k)) for j in (1,2) for k in combinations_with_replacement(range(4),j)]); cfit=fit(aff(power(uc,2)),pc['v1']); print(cfit); c=aff(power(z,2))@cfit
H,A,r=modes(T['traj'][:52]); A=fit(H,T['traj'].T).T; Y=np.c_[T['y'],A]; print('mode rates/correlations',r,np.corrcoef(Y.T)); n=52

def polynames(p,d):return [''.join(str(i) for i in k) for j in range(1,d+1) for k in combinations_with_replacement(range(p),j)]
def run(name,F,N,detail=False):
 mean=F[:n].mean(0); std=np.maximum(1e-10,F[:n].std(0)); X=(F-mean)/std
 for j in range(3):
  yt=Y[:,j]; pars=[.00025,.001,.003,.008,.02]; print(name,'output',j)
  for alpha in pars:
   model=Lasso(alpha=alpha*yt[:n].std(),max_iter=30000,tol=2e-6).fit(X[:n],yt[:n]); yh=model.predict(X); cc=model.coef_/std; inds=np.argsort(abs(model.coef_))[-9:][::-1]; print(alpha, np.round([np.mean(abs(yh[:n]-yt[:n])),np.mean(abs(yh[n:]-yt[n:]))],5),[(N[i],round(cc[i],4)) for i in inds] if detail and alpha in (.001,.008) else '')
run('z cubic',power(z,3),polynames(4,3),True)
run('z quartic',power(z,4),polynames(4,4),True)
run('z+oracle modes',power(np.c_[z,A],2),polynames(6,2),True)
F=[power(z,3)]; N=polynames(4,3)
def add(q,names): F.append(q);N.extend(names)
for tag,q in [('B',b),('C',c),('W',w)]:
 add(power(q,2),[tag+s for s in polynames(4,2)]);add((z[:,:,None]*q[:,None,:]).reshape(-1,16),[f'{i}{tag}{j}' for i in range(4) for j in range(4)])
for tag,q in [('z',z),('b',b),('c',c),('w',w)]:
 for k,fun in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda x:x/(1+x*x))]:
  add(fun(q),[f'{k}({tag}{i})' for i in range(4)])
add((c[:,:,None]*w[:,None,:]).reshape(-1,16),[f'C{i}W{j}' for i in range(4) for j in range(4)])
run('operators',np.column_stack(F),N,True)
