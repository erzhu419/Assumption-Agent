exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
for name,g,z in [('bench',tr,x[0]),('chamber',pi,x[2])]:
 tau,w,_=s._fit(g,z,b,dt,t)
 a=s._design(g,z,b,dt,t,tau); r=s._obs(g)-a@w
 slope=np.median(np.column_stack((r[:,:-1]/t,r[:,-1])),axis=1)
 print(name,'residual drift by lab and stratum:')
 for lab in np.unique(g['lab_id']):
  for st in np.unique(g['stratum']):
   idx=np.flatnonzero((g['lab_id']==lab)&(g['stratum']==st))
   print('cell',lab,st,'slopes',slope[idx].round(4),'X',z[idx].round(2).tolist() if name=='chamber' else '')
 print('abs slope by stratum',[(st,np.median(abs(slope[g['stratum']==st])),np.max(abs(slope[g['stratum']==st]))) for st in np.unique(g['stratum'])])
 if 'aux_chamber_telemetry' in g:
  tele=g['aux_chamber_telemetry']; print('telemetry/feature correlations',np.corrcoef(np.column_stack((tele,z,slope)).T)[:4,4:].round(2))
