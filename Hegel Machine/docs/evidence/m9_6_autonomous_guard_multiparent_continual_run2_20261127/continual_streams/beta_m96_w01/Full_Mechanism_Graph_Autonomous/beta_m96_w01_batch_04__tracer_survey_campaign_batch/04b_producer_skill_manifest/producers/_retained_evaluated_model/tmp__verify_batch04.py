import importlib.util, pickle
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, eye, diags, kron
from scipy.sparse.linalg import splu
p=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',p/'beta_m96_w01_batch_04_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec((p/'validate_public_schema.py').read_text().split('\n_d =')[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
items=pickle.loads((p/'public_beta_m96_w01_batch_04.pkl').read_bytes())

def axis(s,a,f):
    k=f*np.asarray(s['kappa_'+a]); n=len(k); h=s['L'+a]/n
    v=np.broadcast_to(s['vel_'+a],(n,)); R=np.broadcast_to(s['reaction_'+a],(n,)); r=s['robin_'+a]
    A=lil_matrix((n,n))
    for i in range(n):
        A[i,i]=R[i]+abs(v[i])/h
        for j in (i-1,i+1):
            if 0<=j<n:
                d=2/(1/k[i]+1/k[j])/h**2
                A[i,i]+=d
                A[i,j]=-d-(abs(v[i])/h if (j-i)*v[i]<0 else 0)
        if i in (0,n-1): A[i,i]+=r/(h*(1+r*h/(2*k[i])))
    return A.tocsc()

def rel(a,b): return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15)

def compare(s,factor,tag,gt=None):
    src=s['source_modes']; nm,nx,ny=src.shape; ii,jj=s['sensor_ij'].T
    A=kron(axis(s,'x',factor),eye(ny),format='csc')+kron(eye(nx),axis(s,'y',factor),format='csc')
    blocks=[]
    for mu in s['mu_shifts']:
        op=A+diags(s['V_reaction_2d'].ravel()+mu,format='csc'); rhs=src.reshape(nm,-1).T
        u=splu(op).solve(rhs); assert rel(op@u,rhs)<1e-10
        fields=u.T.reshape(src.shape); blocks.append(fields[:,ii,jj].T)
    D=np.concatenate(blocks); dd,ff=m._forward(s)
    de,fe=rel(dd,D),rel(ff,fields); assert max(de,fe)<1e-9,(tag,de,fe)
    out=m.solve_beta_m96_w01_batch_04(s); w=out['weights']; cert=out['kkt_certificate']
    assert np.array_equal(w,cert['weights'])
    term=rel(out['terminal_field'],np.einsum('m,mij->ij',w,fields)); assert term<1e-9
    Z=s['Z_nuisance']; cols=np.column_stack((D,m._calibrate(s)))
    prof=cols-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@cols)
    st,ry=(prof[:,:-1],prof[:,-1]) if gt is None else (gt['S_tilde'],gt['r_y'])
    ck=verify(st,ry,s['source_ridge'],s['u_box'],s['total_budget'],cert)
    print(tag,'design, terminal modes, terminal field errors and KKT:',de,fe,term,ck,flush=True)
    if gt is not None:
        print('public references:',sorted(gt),'profiled design and target errors:',rel(prof[:,:-1],st),rel(prof[:,-1],ry),flush=True)
    assert ck['valid'],ck

for it in items:
    s=it['study']; compare(s,1.45,'public',it['gt'])
s=items[0]['study']; V=s['V_reaction_2d']; c=s['critical_vortex_circulation']; B=V.mean(0)+(V.mean(1)-V.mean())[:,None]
for tag,v,h,f in [('laminar',B,[0,c],1.),('reaction_only',V,[0,c],1.45),('memory_only',B,[0,1.1*c,0],1.45),('both',V,[1.1*c,0],1.45)]:
    compare(dict(s,V_reaction_2d=v,past_vortex_circulation_trace=h),f,tag)
t=s['sensor_calibration_sweeps']; clean=m._calibrate(s); err=0.
for j in range(len(t['bank_id'])):
    k=np.arange(len(t['bank_id']))!=j
    err=max(err,np.max(abs(m._calibrate(dict(s,sensor_calibration_sweeps={a:v[k] for a,v in t.items()}))-clean)))
spike={a:v.copy() for a,v in t.items()}
for b,e in enumerate([1.4,-1.1,1.6,-1.8]):
    j=np.flatnonzero(spike['bank_id']==b); spike['raw_readout'][j[np.argmax(abs(spike['ref_signal'][j]))]]+=e
emi=np.max(abs(m._calibrate(dict(s,sensor_calibration_sweeps=spike))-clean))
print('max calibrated observation differences: leave-one-out',err,'EMI',emi)
assert max(err,emi)<1e-6
