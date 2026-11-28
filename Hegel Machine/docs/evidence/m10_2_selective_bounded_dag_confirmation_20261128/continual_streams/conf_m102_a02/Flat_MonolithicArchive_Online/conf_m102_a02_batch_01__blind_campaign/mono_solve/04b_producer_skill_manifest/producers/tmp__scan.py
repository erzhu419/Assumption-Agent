import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _calibration,_time_basis,solve_batch
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; b,d=_calibration(s['panel_a']); X=(tr['x']-b)/d; z=tr['traj']; t=s['eval_times']; B=_time_basis(t,z[:52]); Y=np.c_[tr['y'],z@np.linalg.pinv(B.T)]; print('calib',b,d,'B',B[:1],flush=True)
P=np.c_[np.ones(len(X)),X]; i,j=np.triu_indices(4,1); cross=X[:,i]*X[:,j]; triple=np.column_stack([X[:,k].prod(1) for k in combinations(range(4),3)])
fn={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'sqrt':lambda a:a/np.sqrt(1+a*a),'rat':lambda a:a/(1+a*a),'rcube':lambda a:a**3/(1+a**6),'rabs':lambda a:a*abs(a)/(1+a*a),'soft':lambda a:a/(1+abs(a)),'sinh':np.sinh}
for label,f in fn.items():
 for mode in [0,1,2,3]:
    base=np.c_[P,*([X*X] if mode&1 else []),*([cross,triple] if mode&2 else [])];
    def des(p): return np.c_[base,f(X*p[:4]+(p[4:] if len(p)==8 else 0))]
    def calc(p):
        A=des(p); c=np.linalg.lstsq(A[:52],Y[:52],rcond=None)[0]; return A@c,c
    def res(p):
        R=(calc(p)[0]-Y)[:52]; return np.c_[R[:,0],R[:,1:]@B.T/np.sqrt(8)].ravel()
    sols=[]
    for shift in [False,True]:
        q=least_squares(res,np.r_[[1.25]*4,[0]*4] if shift else [1.25]*4,bounds=(np.r_[[.06]*4,[-2]*4] if shift else [.06]*4,np.r_[[7]*4,[2]*4] if shift else [7]*4),max_nfev=200)
        r=calc(q.x)[0]-Y; r=np.c_[r[:,0],r[:,1:]@B.T]; sols.append((np.round(q.x,3).tolist(),np.round(np.mean(abs(r[:52]),0)[[0,-1]],4).tolist(),np.round(np.mean(abs(r[52:]),0)[[0,-1]],4).tolist()))
    print(label,mode,sols,flush=True)
