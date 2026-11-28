import importlib.util, pickle
import numpy as np
from scipy.sparse import diags, eye, kron
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
y=m._calibrate(s); Z=s['Z_nuisance']; n=len(s['sensor_ij']); rows=np.arange(len(y)); I=np.eye(Z.shape[1]); cap=s['u_box']; budget=s['total_budget']

def design(code):
    blocks=[]
    for j,axis in enumerate('xy'):
        k=s['kappa_'+axis]; v=s['vel_'+axis]; h=s['L'+axis]/(len(k)-1); r=s['robin_'+axis]
        a=m._axis(k,v,h,r).tolil()
        if code[0] in (j,2):
            diffusion, robin, advection=code[1:]
            for i,b,speed in ((0,1,max(0.,-v[0])),(len(k)-1,len(k)-2,max(0.,v[-1]))):
                flux=diffusion*k[i]/h**2+advection*speed/h
                a[i,i]=flux+robin*r/h; a[i,b]=-flux
        blocks.append(a.tocsc())
    a,b=blocks
    op=kron(eye(b.shape[0]),a,format='csc')+kron(b,eye(a.shape[0]),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    return m._responses(s,op)[0]

base=(-1,1,1,0)
codes=[base]+[(axis,*params) for axis in range(3) for params in [(1,0,0),(1,.5,0),(1,2,0),(.5,1,0),(2,1,0),(2,2,0),(1,1,1)]]
A={code:design(code) for code in codes}

def predict(S,train,test):
    z=Z[train]; ss=S[train]; yy=y[train]
    P=np.linalg.solve(z.T@z+s['nuisance_ridge']*I,z.T)
    T=ss-z@(P@ss); q=yy-z@(P@yy)
    w,_=m._box_simplex(T.T@T+s['source_ridge']*np.eye(T.shape[1]),-T.T@q,cap,budget)
    return S[test]@w+Z[test]@(P@(yy-ss@w))-y[test]

groups=[(rows%n//4)==j for j in range(n//4)]
held={'baseline':np.empty(len(y)),'selected':np.empty(len(y))}; selections=[]
for j,test in enumerate(groups):
    score={c:sum(np.sum(predict(S,~(test|inner),inner)**2) for i,inner in enumerate(groups) if i!=j) for c,S in A.items()}
    selected=min(score,key=score.get); selections.append(selected)
    for label,c in [('baseline',base),('selected',selected)]: held[label][test]=predict(A[c],~test,test)
print('nested wall selection by held-out rosette',selections)
for label,res in held.items():
    rms=lambda r: float(np.sqrt(np.mean(r*r)))
    bank=s['row_bank_id']; print(label,'pooled',rms(res),'worst rosette',max(rms(res[g]) for g in groups),'worst bank',max(rms(res[bank==b]) for b in np.unique(bank)))
print('baseline/challenger shift RMSE for the three best fixed wall candidates (exploratory):')
shift=[rows//n==j for j in range(len(y)//n)]
shift_scores={c:[float(np.sqrt(np.mean(predict(S,~g,g)**2))) for g in shift] for c,S in A.items()}
for c in dict.fromkeys([base]+sorted(codes,key=lambda c:max(shift_scores[c]))[:3]): print(c,shift_scores[c])
