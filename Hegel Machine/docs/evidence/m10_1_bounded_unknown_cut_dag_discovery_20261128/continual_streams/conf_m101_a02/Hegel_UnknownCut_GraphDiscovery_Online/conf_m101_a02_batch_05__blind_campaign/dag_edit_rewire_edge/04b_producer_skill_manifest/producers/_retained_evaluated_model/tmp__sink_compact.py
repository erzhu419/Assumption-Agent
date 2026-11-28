exec(open('/tmp/sink_profile.py').read().split("checks('fixed sin'")[0])
checks('sine, full calibrated linear',lambda tr:(np.c_[base,x,np.sin(w)],[]))
for name,f in [('sine',np.sin(w)-w),('cubic',w**3),('rational',w**3/(1+w*w))]:
 checks(name+' plus x2', lambda tr:(np.c_[base,x[:,2],f],[]))
 checks(name+' plus all x', lambda tr:(np.c_[base,x,f],[]))
 checks(name+' alone', lambda tr:(np.c_[base,f],[]))
# A disjoint ordering by geometric displacement stresses joint coordinate shifts.
r=np.linalg.norm((np.c_[x,z]-np.c_[x,z].mean(axis=0))/np.c_[x,z].std(axis=0),axis=1)
for D in [np.c_[base,x,np.sin(w)],np.c_[base,x[:,2],np.sin(w)-w],np.c_[base,x[:,2],w**3],np.c_[base,w**3]]:
 vals=[]
 for te in np.array_split(np.argsort(r),6):
  tr=np.setdiff1d(ids,te); vals.append(np.mean(abs(D[te]@np.linalg.lstsq(D[tr],y[tr],rcond=None)[0]-y[te])))
 print('radius slices',D.shape[1],np.round(vals,5))
