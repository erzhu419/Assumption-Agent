exec(open('/tmp/explore.py').read().split("p=s['panel_c'];")[0])
p=s['panel_c']; a,ids=terms(p['u'],2)
for k in ['v1','v2']:
 print(k)
 for cols in [list(range(15)), [i for i,j in enumerate(ids) if len(set(j))<=1], [i for i,j in enumerate(ids) if len(set(j))<=1 and len(j)<2 or len(set(j))==2]]:
  g=a[:,cols]; w=np.linalg.lstsq(g,p[k],rcond=None)[0]; print('rms',np.sqrt(np.mean((g@w-p[k])**2,0))); print(list(zip([ids[i] for i in cols],w.round(4))))
pa=s['panel_a']; z2,_=old._coordinates(s,pa['v']);print('A calibration residual',np.quantile(abs(z2-pa['u']),[.3,.5,.8,.9,.95,.99]));print('A fit',np.linalg.lstsq(f(pa['v'][np.max(abs(z2-pa['u']),axis=1)<.01]),pa['u'][np.max(abs(z2-pa['u']),axis=1)<.01],rcond=None)[0])
