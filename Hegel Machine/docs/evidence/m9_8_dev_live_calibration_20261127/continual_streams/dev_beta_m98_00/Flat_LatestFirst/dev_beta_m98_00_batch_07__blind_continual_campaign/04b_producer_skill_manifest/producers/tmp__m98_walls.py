import importlib.util,itertools,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

def load(b):
    path=f'/tmp/hegel_continual_archive/dev_beta_m98_00_batch_{b}_solver.py'
    spec=importlib.util.spec_from_file_location('m'+b,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

m,p=load('06'),load('03')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
W=np.column_stack([d['w_pulse'] for d in item['campaign_diagnostics']]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in item['campaign_diagnostics']]); rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']])@W
reaction=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); eye=sparse.eye(nx*ny,format='csc')

def predict(ax,ay):
    op=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+reaction
    return np.vstack([splu(op+mu*eye).solve(rhs)[rows] for mu in s['mu_shifts']])

def axes(a):
    k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]/h
    ref=m._axis(k,v,h).toarray(); per=p._axis(k,v,h)
    out={'periodic':per}
    for scale,leak in itertools.product([1,2],[0,0.25,0.5,1,2]):
        mat=ref.copy(); mat[[0,-1]]*=scale; mat[0,0]+=leak*r; mat[-1,-1]+=leak*r
        out[f'diff{scale}_robin{leak}']=sparse.csc_matrix(mat)
    for name,sc in [('dirichlet',1e5),('identity',1.)]:
        mat=ref.copy(); mat[[0,-1]]=0; mat[0,0]=mat[-1,-1]=sc; out[name]=sparse.csc_matrix(mat)
    return out

if __name__=='__main__':
    scored=[]
    for (a,ax),(b,ay) in itertools.product(axes('x').items(),axes('y').items()):
        e=(predict(ax,ay)-Y).reshape(3,-1,2); y=Y.reshape(3,-1,2)
        rel=np.linalg.norm(e,axis=1)/np.linalg.norm(y,axis=1); scored.append((rel.max(),np.mean(e**2)**.5,a,b,rel))
    for worst,rmse,a,b,rel in sorted(scored,key=lambda t:t[0])[:8]: print(a,b,'max rel',round(worst,7),'RMSE',round(rmse,7),'by shift and pulse',np.round(rel,7).tolist())
    for name in ['periodic','diff1_robin0','diff1_robin1']:
        print('both',name,next(row[:2] for row in scored if row[2:4]==(name,name)))
