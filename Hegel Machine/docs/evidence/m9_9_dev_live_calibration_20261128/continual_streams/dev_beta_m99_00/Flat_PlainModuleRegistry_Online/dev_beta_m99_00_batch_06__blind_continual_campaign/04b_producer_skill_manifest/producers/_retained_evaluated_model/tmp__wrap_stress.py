exec(open('/tmp/compare_chamber.py').read().split('for x,b in itertools.product')[0])
# The prefix record names the completed campaign, so its setting applies to the next one.
logs=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
known={'facility_init':'RR','dev_beta_m99_00_s00':'SS','dev_beta_m99_00_batch_01':'PP','dev_beta_m99_00_batch_04':'PS'}
latest=np.asarray(logs[-1]['facility_telemetry_trace'])
for row in logs:
 name=row['completed_campaign_short_name']
 if name in known:
  a=np.asarray(row['facility_telemetry_trace']); print('prior setting',known[name], 'mean-signature distance',round(np.linalg.norm(a.mean(0)-latest.mean(0)),4))
a,b=[np.asarray(logs[i]['facility_telemetry_trace']) for i in (2,5)]; print('PP and PS prefix traces exactly equal:',np.array_equal(a,b))
# Remove bank offsets fit only on training data. Keep the specified nuisance projection,
# and compare with the unmodified nuisance fit, holding complete rosettes together.
rosette=np.tile(np.arange(ns)//4,nr//ns); banks=s['row_bank_id']; idx=np.arange(nr)
for extra in (False,True):
 B=np.column_stack((Z,np.eye(4)[banks])) if extra and np.array_equal(np.unique(banks),np.arange(4)) else (np.column_stack((Z,np.column_stack([banks==g for g in np.unique(banks)]))) if extra else Z)
 for kind in ('RR','SS','SP','PS','PP'):
  x,b=kind; L=(kron(eye(ny),axis('x',nx,x))+kron(axis('y',ny,b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); S=np.vstack([splu(L+mu*I).solve(rhs)[pos] for mu in s['mu_shifts']]); summary=[]
  for groups in (idx//ns,rosette):
   E=np.zeros(nr)
   for g in np.unique(groups):
    tr=groups!=g; A,C,y1=S[tr],B[tr],y[tr]; P=np.linalg.solve(C.T@C+ridge*np.eye(C.shape[1]),C.T); A1=A-C@(P@A); t=y1-C@(P@y1); w,_=old._box_simplex(A1.T@A1+sr*np.eye(A.shape[1]),-A1.T@t,cap,total); E[~tr]=(S@w+B@(P@(y1-A@w))-y)[~tr]
   summary.append(tuple(round(float(q),6) for q in [rmse(E),max(rmse(E[groups==g]) for g in np.unique(groups)),max(rmse(E[banks==g]) for g in np.unique(banks))]))
  print('bank-offset audit',extra,kind,'shift/rosette pooled, worst fold, worst bank',summary)
