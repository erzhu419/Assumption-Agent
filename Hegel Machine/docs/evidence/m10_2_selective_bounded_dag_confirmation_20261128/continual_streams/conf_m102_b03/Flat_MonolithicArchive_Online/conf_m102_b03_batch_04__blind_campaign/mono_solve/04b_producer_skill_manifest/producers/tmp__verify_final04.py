exec(open('/tmp/validate_batch04.py').read().split('records=[]')[0])
records=[]
for name,i,j in splits:
    q={**s,'train':{**train,**{k:train[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}
    out=m.solve_batch(q); yy=out['y_hat']; state=out['state_pred']; last=state[:,-1]
    assert yy.shape == (len(j),) and state.shape == (len(j),t.shape[1])
    np.testing.assert_allclose(out['diagnostics_vec'],[yy.mean(),yy.std(),last.mean(),last.std()]); assert all(np.isfinite(v).all() for v in out.values())
    ey=abs(yy-y[j]); et=abs(state-t[j]); records.append((name,ey.mean(),et.mean(),et.mean(0).max(),ey.max(),et.max()))
print('official:',records[0]);print('worst across',len(records),'splits:',*[max(records,key=lambda r:r[k]) for k in range(1,6)],sep='\n');assert max(max(r[1:4]) for r in records)<.038
out=m.solve_batch(s); yy=out['y_hat']; state=out['state_pred']; last=state[:,-1]
assert yy.shape==(36,) and state.shape==(36,8) and all(np.isfinite(v).all() for v in out.values())
np.testing.assert_allclose(out['diagnostics_vec'],[yy.mean(),yy.std(),last.mean(),last.std()]); print('diagnostics:',out['diagnostics_vec'])
p=s['panel_c']; u=p['u']; v=p['v1']; n=len(u)
for name,f in [('coordinatewise',np.c_[u,u*u]),('cyclic',u*np.roll(u,-1,axis=1))]:
    d=m._design(f); err=[]
    for j in range(n):
        i=np.arange(n)!=j; err.extend(d[j] @ m._ls(d[i],v[i])-v[j])
    print('panel_c.v1 held-out MAE:',name,np.mean(np.abs(err)))
    if name=='cyclic': assert np.mean(np.abs(err)) < .001
