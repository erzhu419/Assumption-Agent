exec(open('/tmp/compact14.py').read().split('for mode in range(4)')[0])
fast=[1,2,3,4,9]; slow=[5,6,7,8,10]
for mode in range(3):
 metrics=[]; rates=[]
 for va in idxs:
  tr=np.setdiff1d(np.arange(72),va)
  def design(ids,r):
   b=basis(r); cols=[(D[ids,i,None]*b[None,:,1 if i in fast else 0]).ravel() for i in fast+slow];
   if mode==1: cols.extend(np.broadcast_to(b[None,:,j],(len(ids),len(t))).ravel() for j in range(2))
   if mode==2: cols.extend((D[ids,i,None]*b[None,:,1]).ravel() for i in slow)
   return np.column_stack(cols)
  target=tt[tr].ravel()
  def res(r):
   F=design(tr,r);return F@m._ls(F,target)-target
  r=least_squares(res,[.4,1.8],bounds=([.03,.9],[.89,4.]),gtol=1e-11).x
  st=(design(va,r)@m._ls(design(tr,r),target)).reshape(-1,len(t));e=abs(st-tt[va]);metrics.append([e.mean(),e.mean(0).max(),e.max()]);rates.append(r)
 print(mode,'rates',rates[0],'fixed',metrics[0],'worst',np.max(metrics,0))
