from node03_calibration import *
from pathlib import Path
p=Path('/tmp/node03_loo.pkl')
if not p.exists():
 out=[]
 for i,v in enumerate(np.array_split(np.random.default_rng(198).permutation(dev),4)+[official]):
  tr=np.setdiff1d(dev,v)
  out.append(([build(tr[tr!=j],[j]) for j in tr],build(tr,v)))
 pickle.dump(out,p.open('wb'))
R=pickle.load(p.open('rb'))
for fold,(tr,v) in enumerate(R):
 print('fold',fold)
 for rs in (tr,[v]):
  A=np.vstack([np.column_stack((p[:,0],b[:,0],b[:,1:]@vt,(p[:,1:]-b[:,1:])@vt,y)) for p,b,vt,y,_ in rs]);p,b,*_,y=A.T; d=p-b; r=y-b
  print('mae',np.mean(abs(r)),'corr',np.round(np.corrcoef(np.column_stack((A[:,:2],A[:,2],A[:,9],A[:,10],A[:,17],np.abs(d),np.sign(d)*np.sqrt(abs(d)),b*b,b**3,d,r)).T)[-1,:-1],3)); print('cond',[(int(m.sum()),round(np.mean(r[m]),4),round(np.mean(abs(r[m])),4)) for m in (d<-.02,abs(d)<=.02,d>.02,b<-.6,b>.6)])
