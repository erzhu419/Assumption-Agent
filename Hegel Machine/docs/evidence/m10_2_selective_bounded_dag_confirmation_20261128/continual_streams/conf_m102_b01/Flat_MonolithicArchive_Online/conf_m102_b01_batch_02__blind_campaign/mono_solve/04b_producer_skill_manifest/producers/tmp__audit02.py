import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj')); ix=np.arange(len(x)); z=m._calibrate(s['panel_a'],x); b=s['panel_b']; w=m._aff(z)@m._fit(m._aff(b['u']),b['v']); h=m._latent(s,x); F=m._aff(h); S=np.c_[F,h[:,0]*h[:,2],h[:,1]*h[:,3]]; pairs=[(0,1),(0,3),(1,2),(2,3)]; D=np.column_stack([((w[:,i]+w[:,j])**2-(w[:,i]-w[:,j])**2)/4 for i,j in pairs]); print('signed-square identity max error',np.max(abs(D-m._products(w)))); print('study keys',sorted(s),'train keys',sorted(tr),'telemetry shape',np.shape(s['telemetry_vec']))
Q=np.c_[h,w,*[(h[:,i]-h[:,j])[:,None] for i,j in pairs]]
bank=[(j,a,1+a*(Q[:,j]/max(np.std(Q[:,j]),1e-9))**2) for j in range(Q.shape[1]) for a in (.0005,.002,.01,.05,.2)]
def pred(i,j,X,Y,rank=False):
    B=m._fit(X[i],Y[i]); p=X[j]@B
    if rank:
        V=np.linalg.svd(X[i]@B,full_matrices=False)[2][:2]; p=(p@V.T)@V
    return p
def mae(e):return np.mean(abs(e))
def rational(i,j,X,Y,rank):
    splits=np.array_split(np.random.default_rng(74).permutation(i),4)
    best=(float('inf'),None,None)
    for col,a,d in bank:
        xx=X/d[:,None]; v=np.mean([mae(pred(np.setdiff1d(i,k),k,xx,Y,rank)-Y[k]) for k in splits]);
        if v<best[0]: best=(v,col,a); B=xx
    return pred(i,j,B,Y,rank),best[1:]
C=s['panel_c']; W=m._fit(m._aff(m._products(C['u'])),C['v1']); raw=m._aff(m._products(z))@W
full2=np.c_[m._aff(z),*[z[:,a]*z[:,b] for a in range(4) for b in range(a,4)]]
trip=np.column_stack([h[:,a]*h[:,b]*h[:,c] for a,b,c in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]);
rows=[]
for name,j in [('public',ix[52:]),('early',ix[:20]),('high_input',np.argsort(np.linalg.norm(z,axis=1))[-20:]),('high_latent',np.argsort(np.linalg.norm(h,axis=1))[-20:]),('low_y',np.argsort(y)[:20]),('high_y',np.argsort(y)[-20:])]:
    i=np.setdiff1d(ix,j); a=m.solve_batch(dict(s,train={'x':x[i],'y':y[i],'traj':T[i]},test={'x':x[j]})); ry,sy=rational(i,j,S,y,False); rt,st=rational(i,j,F,T,True)
    e=a['state_pred']-T[j]; row=[mae(a['y_hat']-y[j]),mae(e),max(np.mean(abs(e),axis=0)),mae(ry-y[j]),mae(rt-T[j]),mae(pred(i,j,np.c_[S,trip],y)-y[j]),mae(pred(i,j,full2,T,True)-T[j]),mae(pred(i,j,m._aff(raw),T,True)-T[j])]; rows.append(row); print(name,*[f'{r:.6f}' for r in row], 'rational',sy,st)
print('columns: scalar trajectory worst_horizon rational_scalar rational_trajectory triple_scalar unrestricted_quadratic_trajectory omit_B_trajectory'); print('maxima',np.max(rows,axis=0))
a=m.solve_batch(s); np.testing.assert_array_equal(a['diagnostics_vec'],[a['y_hat'].mean(),a['y_hat'].std(),a['state_pred'][:,-1].mean(),a['state_pred'][:,-1].std()]); assert np.max(rows,axis=0)[:3].max()<.038
