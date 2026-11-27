exec(open('/tmp/rheo_factor.py').read().split("for mode,spec in")[0])
for mode,spec in [('load','fixed'),('cell','fixed'),('load','soft')]:
 pars=factor(allix,mode,spec); print(mode,spec,'drift coefficients',np.round(pars[2]*np.r_[[.1]+[.08]*4+[.12]*12, ([.045]+[.035]*12) if mode=='load' else []],4));
 for tags in (np.arange(len(cell)),cell,pi['lab_id']):
  yp=np.zeros_like(s._obs(pi))
  for k in np.unique(tags):
   ix=tags==k; yp[ix]=pred({key:v[ix] for key,v in pi.items()},x[2][ix],factor(~ix,mode,spec),mode)
  r=abs(yp-s._obs(pi)); c=r[:,:-1]; v=r[:,-1]
  print('folds',len(np.unique(tags)),'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'worst cell',np.round([max(c[cell==i].mean() for i in np.unique(cell)),max(v[cell==i].mean() for i in np.unique(cell))],4))
train_hold=tr['hold_channel_raw']@b[1:]+b[0]
for name,g in [('pilot',pi),('test',te)]:
 hold=g['hold_channel_raw']@b[1:]+b[0]; c=g['lab_id']*3+g['stratum']; print(name,'hold support',[(int(i),np.round([min(hold[c==i]),max(hold[c==i])],2).tolist()) for i in np.unique(c)])
print('train hold median',np.median(train_hold))
