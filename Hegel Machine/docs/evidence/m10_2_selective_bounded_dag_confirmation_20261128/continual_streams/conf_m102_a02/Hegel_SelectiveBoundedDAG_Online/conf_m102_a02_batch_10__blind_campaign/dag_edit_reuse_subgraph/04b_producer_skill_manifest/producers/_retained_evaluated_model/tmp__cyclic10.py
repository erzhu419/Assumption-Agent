exec(open('/tmp/ex10.py').read().split("print('operator'")[0] if False else open('/tmp/ex10.py').read().split('for name,a,c')[0].replace("; print('operator',o)",""))
a8=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a02_batch_08_solver.py'); pa=s['panel_a']; co=a8['_fit_calibration'](pa); print('calibration residual quantiles',*[np.quantile(np.abs(r),[.4,.7,.8,.9,.95,1.]) for r in [pa['u']*o['gain']+o['offset']-pa['v'],a8['_design'](pa['u'],len(co))@co-pa['v']]]); print('q percentiles',np.quantile(q,[0,.01,.5,.99,1]));
for name,Q in [('frozen',q),('arch',a8['_invert'](s['train']['x'],co))]:
 G=b(Q); print('---',name)
 for j in range(4):
  c=G*np.roll(Q,-j,axis=1); gate=1/(1+s['telemetry_vec'][2]*Q[:,2]**2)
  for k,z in [('base',f(G,c.sum(axis=1))),('cyc',f(G,c)),('qcyc',f(Q,G,c)),('gated',f(G*gate[:,None],c,G[:,3]*Q[:,2]**3*gate))]:
   report(f'{j}{k} traject',z,t); report(f'{j}{k} scalar',z,y)
 for d in [1,2]:
  A=f(u,bu) if d==1 else f(u,bu,pairs(u)); X=f(Q,G) if d==1 else f(Q,G,pairs(Q)); V=X@ls(A,np.c_[v1,v2]);
  for k,z in [('C',f(V)),('CB',f(G,V)),('CBcyc',f(G,V,G*np.roll(Q,-1,axis=1)))]:
   report(f'{d}{k} traject',z,t);report(f'{d}{k} scalar',z,y)
print('archive end-to-end'); ss={**s,'train':{k:v[:n] for k,v in s['train'].items()},'test':{'x':s['train']['x'][n:]}}; out=a8['solve_batch'](ss); print(np.mean(abs(out['y_hat']-y[n:])),np.mean(abs(out['state_pred']-t[n:])))
