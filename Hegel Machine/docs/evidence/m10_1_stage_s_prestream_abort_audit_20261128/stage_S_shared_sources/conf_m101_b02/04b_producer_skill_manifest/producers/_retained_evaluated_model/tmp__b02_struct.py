import importlib.util,pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
sp=importlib.util.spec_from_file_location('base','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def kernels(z):
    groups=np.split(z,3,axis=1); ds=[cdist(g,g,'sqeuclidean')/4 for g in groups]; gs=[g@g.T/4 for g in groups]
    for h in (.25,.55,1.1):
        rb=[np.exp(-h*d) for d in ds]; dr=np.sqrt(h*sum(ds)/3); full=(1+dr+dr**2/3)*np.exp(-dr)
        for w in (0,.25,1):
            for p in (0,.06,.25):
                yield ('group',h,w,p), full+w*(sum(rb)/3)+p*sum(gs)/3
        U=np.exp(-h*(z[:,None,:]-z[None,:,:])**2); a=U.mean(2); b=(U.sum(2)**2-(U*U).sum(2))/132
        for w in (.25,1.):
            yield ('component',h,w), full+w*(.5*a+.5*b)
    for h in (.035,.1,.25):
        for w in (.03,.12,.4):
            G=sum(gs)/3
            yield ('poly',h,w), (1+G)**2+w*sum(np.exp(-h*d) for d in ds)/3+w*np.exp(-h*sum(ds)/3)

def eval_model(study, report=False):
    x=study['train']['x']; y=study['train']['y']; T=study['train']['traj']; F=m._basis(study['eval_times'],T); Y=np.c_[y,lstsq(F,T.T)[0].T]
    n=len(x); rows=np.arange(n); te=np.arange(n,n+len(study['test']['x'])); Z=m._coordinates(study,np.r_[x,study['test']['x']])
    def norm(z):return (z-z[:n].mean(0))/z[:n].std(0)
    z=norm(Z['abcd']); raw=norm(Z['raw']); rad=(raw[:n]**2).sum(1)
    folds=[*np.array_split(np.random.default_rng(498).permutation(n),4), np.argsort(rad)[-max(6,n//7):]]
    splits=[(np.setdiff1d(rows,j),j) for j in folds]
    records=[]
    for k,K in [*kernels(z), *[(('raw',k),K) for k,K in m._kernels(raw)], *[(('abcd',k),K) for k,K in m._kernels(z)]]:
        if k[0] in ('raw','abcd') and k[1].startswith('anova'):continue
        for reg in (1e-6,3e-5,.0005,.003,.015):
            e=[]
            for I,J in splits:
                pred=m._pred(K,I,J,Y,reg); d=abs(pred[:,1:]@F.T-T[J]);e.append([abs(pred[:,0]-y[J]).mean(),.4*d.mean()+.6*d[:,-1].mean()])
            e=np.array(e); score=.50*e[:-1].mean(0)+.15*e[:-1].max(0)+.35*e[-1]
            records.append((score,k,reg,e,K))
    result=[]
    for j in range(2):
        best=sorted(records,key=lambda a:a[0][j]); used=set(); ps=[]; ws=[]
        if report:print('inner',j,[(k,l,np.round(e[:,j],4)) for _,k,l,e,_ in best[:3]])
        for score,k,l,e,K in best:
            if k[0] in used:continue
            q=score[j]/max(best[0][0][j],1e-10)
            if q>1.23:break
            used.add(k[0]); ps.append(m._pred(K,rows,te,Y,l));ws.append(np.exp(-7*(q-1)))
        p=np.average(ps,axis=0,weights=ws);result.append(p[:,0] if j==0 else p[:,1:]@F.T)
    return result
if __name__=='__main__':
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y'];T=s['train']['traj'];n=len(x);a=np.arange(n);r=np.sum(((x-x.mean(0))/x.std(0))**2,axis=1)
    z=s['test']['x'];rt=np.sum(((z-x.mean(0))/x.std(0))**2,axis=1);print('radial quantiles train/test',np.quantile(r,[.5,.85,1]),np.quantile(rt,[.5,.85,1]))
    for J in [a[56:],*np.array_split(np.random.default_rng(34).permutation(n),4),np.argsort(r)[-12:]]:
        I=np.setdiff1d(a,J);sub={**s,'train':{k:v[I] for k,v in s['train'].items() if k!='summary'},'test':{'x':x[J]}}
        p,t=eval_model(sub,True);d=abs(t-T[J]);print('errors',np.round([abs(p-y[J]).mean(),d.mean(),d[:,-1].mean(),d.mean(0).max()],6))
