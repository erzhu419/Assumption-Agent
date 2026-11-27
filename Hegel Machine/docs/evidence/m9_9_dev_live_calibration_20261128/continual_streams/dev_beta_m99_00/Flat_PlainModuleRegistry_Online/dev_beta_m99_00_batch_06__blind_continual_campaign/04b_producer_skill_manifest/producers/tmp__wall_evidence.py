exec(open('/tmp/compare_chamber.py').read().split('for x,b in itertools.product')[0])
logs=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); a=np.array([v['facility_telemetry_trace'] for v in logs]);
print('trace fingerprints: index, distance to latest by channel, centered std')
for v,t in zip(logs,a):
 print(v['completed_campaign_index'],np.round(np.sqrt(np.mean((t-a[-1])**2,axis=0)),3),np.round(t.std(axis=0),3))
from scipy.optimize import minimize
fields={}
for x,b in itertools.product('SP',repeat=2):
 L=(kron(eye(ny),axis('x',nx,x))+kron(axis('y',ny,b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); us=[splu(L+mu*I).solve(rhs) for mu in s['mu_shifts']]; S=np.vstack([u[pos] for u in us]); allrows=np.ones(nr,dtype=bool)
 a0,y0,w,gamma=fit(S,allrows); fields[x+b]=us[-1]@w
 print(x+b,'mandated w',np.round(w,5),'gamma',np.round(gamma,5))
 for rho in (0., .005, .04):
  C=np.linalg.lstsq(Z,np.column_stack((S,y)),rcond=None)[0]; B=S-Z@C[:,:-1]; t=y-Z@C[:,-1]; v,nu=old._box_simplex(B.T@B+rho*np.eye(S.shape[1]),-B.T@t,cap,total)
  print('  source ridge',rho,'RSS',np.sum((B@v-t)**2),'w',np.round(v,4))
for k,v in fields.items(): print(k,'terminal relative difference vs PS',np.linalg.norm(v-fields['PS'])/np.linalg.norm(fields['PS']))
