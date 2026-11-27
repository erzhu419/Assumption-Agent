exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0].replace("print(tag,'telemetry quantiles',np.quantile(a,[0,.1,.5,.9,1],axis=0),'first rows',a[:4]); print('by lab means',np.array([a[s['lab_id']==g].mean(0) for g in np.unique(s['lab_id'])]))",'pass'))
l=ch['lab_id']; H=np.eye(4)[l]; r=xx@b[3:]; N=len(r)
for name,A in [('lab gain and offset',np.c_[H,H*r[:,None]]),('lab offsets, common gain',np.c_[H,r]),('lab offsets, free features',np.c_[H,xx]),('lab offsets and gain, cubic features',np.c_[H,H*r[:,None],xx**3@b[3:]]),('quartic scalar drift',np.column_stack([r**i for i in range(5)]))]:
 c=np.linalg.lstsq(A,target,rcond=None)[0]; e=A@c-target; print(name, np.round(c,5),np.round(np.sqrt(np.mean(e**2)),5), np.round(e,4))
for i in np.argsort(r): print(int(i),l[i],np.round(np.r_[r[i]+b[2],target[i],xx[i]],4))
