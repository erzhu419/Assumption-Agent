import numpy as np,pickle,runpy,itertools
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]
pairs=list(itertools.combinations_with_replacement(range(w.shape[1]),2)); F=np.column_stack([np.ones(len(w)),w,*[w[:,i]*w[:,j] for i,j in pairs]]); names=['1']+[str(i) for i in range(w.shape[1])]+[str(p) for p in pairs]
ls=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
def sel(a,Y,penalty):
    J=list(range(F.shape[1])); n=len(a)
    def fit(j):
        A=F[np.ix_(a,j)]; q=ls(A,Y[a]); return n*np.log(max(np.mean((A@q-Y[a])**2),1e-30))+penalty*len(j)*np.log(n)
    while len(J)>1:
        score=fit(J); v,j=min((fit([l for l in J if l!=k]),k) for k in J)
        if v>=score:break
        J.remove(j)
    return J
print('coefficients: term scalar endpoints'); print([(nm,round(v,4),np.round(r[[0,-1]],4).tolist()) for nm,v,r in zip(names,ls(F,y),ls(F,T))]); print('w columns',w.shape)
splits=[(np.arange(52),np.arange(52,72))];
for seed in range(6):splits+=list(KFold(6,shuffle=True,random_state=seed).split(F))
r=np.linalg.norm(w,axis=1); splits += [(np.argsort(r)[:-12],np.argsort(r)[-12:])]
for p in [0,.5,1,2,3,4]:
    for Y in [y,T]:
        err=[]; tails=[]; selection=[]
        for a,b in splits:
            j=sel(a,Y,p); pred=F[np.ix_(b,j)]@ls(F[np.ix_(a,j)],Y[a]); ee=np.abs(pred-Y[b]); err.append(ee.mean()); tails.append(max(ee.mean(axis=0).ravel())); selection.append(j)
        print('p',p,'shape',Y.shape,'public mean max extrap',np.round([err[0],np.mean(err[1:-1]),max(err[1:-1]),err[-1]],6),'worst horizon',round(max(tails),6),'public selected',[names[k] for k in selection[0]],'full selected',[names[k] for k in sel(np.arange(len(y)),Y,p)])
