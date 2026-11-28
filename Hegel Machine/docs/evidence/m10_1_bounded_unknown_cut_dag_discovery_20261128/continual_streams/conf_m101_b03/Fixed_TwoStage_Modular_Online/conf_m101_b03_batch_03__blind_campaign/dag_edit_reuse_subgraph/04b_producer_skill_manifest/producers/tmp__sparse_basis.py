import numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.optimize import minimize_scalar


def panel_features(panel, z):
    u = panel['u']
    v = panel['v1']
    def design(x, f):
        return np.c_[np.ones(len(x)), x, np.sin(f*x)]
    def error(f):
        d = design(u, f)
        return np.linalg.norm(d @ np.linalg.lstsq(d, v, rcond=None)[0] - v)
    f = minimize_scalar(error, bounds=(.5, 4), method='bounded').x
    return (design(z, f) @ np.linalg.lstsq(design(u, f), v, rcond=None)[0]), f, error(f)


def library(z, B, V, C):
    output = {}
    def add(s, x):
        for j, col in enumerate(np.asarray(x).reshape(len(z), -1).T):
            output[(s, j)] = col
    for tag, x in [('z', z), ('B', B), ('V', V), ('C', C)]:
        add(tag, x)
    for tag, x in [('z', z), ('B', B), ('V', V), ('C', C)]:
        for i in range(4):
            a = x[:, i]
            if tag == 'z':
                for k in (2,3,4): add((tag,i,'pow',k), a**k)
            for k in (.7,1.,1.2,1.4,1.5,1.6,1.8,2.,2.2,2.4,2.5,2.7,2.8,3.,3.2,3.5,4.):
                for f in (np.sin,np.cos,np.tanh): add((tag,i,f.__name__,k),f(k*a))
                if tag=='z':
                    add((tag,i,'inv',k),1/(1+k*a*a)); add((tag,i,'rat',k),a/(1+k*a*a))
        for i,j in combinations(range(4),2):
            a,b=x[:,i],x[:,j]; w=a*b
            add((tag,i,j,'prod'),w)
            if tag!='z': continue
            for k in (.5,.7,.8,1.,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2.,2.2,2.4,2.5,2.7,3.,3.5,4.):
                for q, qn in ((w,'prod'),(a+b,'sum'),(a-b,'diff')):
                    for f in (np.sin,np.cos,np.tanh): add((tag,i,j,qn,f.__name__,k),f(k*q))
                    add((tag,i,j,qn,'inv',k), 1/(1+k*q*q)); add((tag,i,j,qn,'rat',k), q/(1+k*q*q))
                for f in (np.sin,np.cos,np.tanh):
                    add((tag,i,j,'int1',f.__name__,k), b*f(k*a)); add((tag,i,j,'int2',f.__name__,k),a*f(k*b))
                add((tag,i,j,'r1',k),a/(1+k*b*b)); add((tag,i,j,'r2',k),b/(1+k*a*a))
    for degree in (3,4):
        for p in combinations_with_replacement(range(4),degree):
            if len(set(p))>1: add(('poly',p),np.prod(z[:,p],axis=1))
    return list(output),np.column_stack(list(output.values()))


def path(X, y, max_terms=15):
    # Forward selection with coordinatewise replacement of previously chosen atoms.
    ids=[]
    def best(ids):
        q=np.linalg.qr(X[:,ids],mode='reduced')[0]; R=q.T@X
        denom=np.sqrt(np.maximum(0,1-np.sum(R*R,axis=0)))
        score=abs(y@X-(y@q)@R)/np.maximum(denom,1e-7)
        score[denom<1e-5]=-1
        return int(score.argmax())
    for _ in range(max_terms):
        ids.append(best(ids))
        for _ in range(8):
            before=ids.copy()
            for j in range(len(ids)):
                ids[j]=best(ids[:j]+ids[j+1:])
            if ids==before: break
        d=X[:,ids]; b=np.linalg.lstsq(d,y,rcond=None)[0]; yield ids.copy(), b, y-d@b

if __name__=='__main__':
    import pickle,runpy
    P='/holdout_task/'; s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; n=runpy.run_path(P+'node_00.py'); sink=runpy.run_path(P+'_frozen_nodes/node_01.py'); z=n['run_node']({'__input__':s})['z'][:72,:4]; C=n['_panel_c_v2'](s['panel_c'],z); V,f,e=panel_features(s['panel_c'],z); p=s['panel_b']; B=np.c_[np.ones(len(z)),z]@np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; print('v1 fit',f,e,flush=True)
    names,D=library(z,B,V,C); y=s['train']['y']; t=s['eval_times']; T=s['train']['traj']; print('dictionary',D.shape,flush=True)
    for a in [np.arange(52),np.arange(72)]:
        b=np.arange(52,72) if len(a)==52 else a
        h,m=sink['_modes'](t,T[a]); scale=np.linalg.norm(D[a]-D[a].mean(0),axis=0); X=(D-D[a].mean(0))/np.maximum(scale,1e-9)
        for q,Y in enumerate(np.c_[y[a],m].T):
            print('target',len(a),q,flush=True)
            for j,(idx,c,r) in enumerate(path(X[a],Y-Y.mean(),18)):
                if j in (3,6,9,12,15,17):
                    ref=y[b] if q==0 else np.linalg.lstsq(h,T[b].T,rcond=None)[0][q-1]; print(j+1,round(np.mean(abs(r)),5),round(np.mean(abs(X[b][:,idx]@c+Y.mean()-ref)),5),[names[i] for i in idx],flush=True)
    try:
        from sklearn.linear_model import LassoLars, ARDRegression
    except ImportError: pass
    else:
        for a,b in [(np.arange(52),np.arange(52,72)),(np.arange(18,72),np.arange(18))]:
            h,m=sink['_modes'](t,T[a]); X=(D-D[a].mean(0))/np.maximum(D[a].std(0),1e-9)
            for al in (.001,.003,.008,.02,.04):
                pred=np.column_stack([LassoLars(alpha=al).fit(X[a],Y).predict(X[b]) for Y in np.c_[y[a],m].T]); print('lasso',a[0],al,np.mean(abs(pred[:,0]-y[b])),np.mean(abs(pred[:,1:]@h.T-T[b])),flush=True)
