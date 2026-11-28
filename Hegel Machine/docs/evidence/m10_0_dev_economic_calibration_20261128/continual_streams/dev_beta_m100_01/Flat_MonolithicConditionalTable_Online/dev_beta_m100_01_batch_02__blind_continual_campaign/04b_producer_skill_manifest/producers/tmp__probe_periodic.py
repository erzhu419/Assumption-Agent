from probe_walls import *
from itertools import product
qfit=np.array([1.03770,0,1.14551,1.03774,0,1.08555,0.99557,0,0.78345,1.0078,0,1.51309]).reshape(4,3)
from scipy.stats import linregress
cal=s['sensor_calibration_sweeps']; a,b=cal['ref_rosette'],cal['raw_rosette']; sb=cal['single_bank_sweeps']; slope=[]; bias=[]
for j in range(4):
 mask=sb['bank_id']==j
 fit=linregress(np.r_[a[:,j],sb['ref_signal'][mask]], np.r_[b[:,j],sb['raw_readout'][mask]])
 slope.append(fit.slope);bias.append(fit.intercept)
Y=np.stack([((e['y_pulse_raw'].reshape(-1,4)-bias)/slope).ravel() for e in p],axis=1)
rows=[]
for k,ab in enumerate('xy'):
 kap=s['kappa_'+ab];v=s['vel_'+ab];h=s['L'+ab]/(len(kap)-1)
 for j,(i,adj,opp) in enumerate(((0,1,-1),(-1,-2,0))):
  t=2*j-1; w=kap[i]/h**2; local=qfit[2*k+j,0]*w; wrap=qfit[2*k+j,2]*w
  avgl=(kap[i]+kap[adj])/(2*h**2);avgw=(kap[i]+kap[opp])/(2*h**2)
  al=max(t*v[i],0)/h;aw=max(-t*v[i],0)/h
  print(ab,i,'local and wrap',local,wrap,'face averages',avgl,avgw,'upwind increments',al,aw,'periodic residual',local-avgl-al,wrap-avgw-aw)
  rows.append((w,avgl,avgw,al,aw))
results=[]
for dl,dw,ul,uw in product((0,1),repeat=4):
 q=[]
 for w,l,r,a,b in rows:q.extend([(l if dl else w)/w + ul*a/w,0,(r if dw else w)/w + uw*b/w])
 from scipy.sparse.linalg import splu
 axes=[]
 for k,ab in enumerate('xy'):
  kap=s['kappa_'+ab];h=s['L'+ab]/(len(kap)-1);A=old._axis(kap,s['vel_'+ab],h).tolil()
  for j,(i,adj,opp) in enumerate(((0,1,-1),(-1,-2,0))):
   l,_,r=q[6*k+3*j:6*k+3*j+3];w=kap[i]/h**2;A[i,i]=(l+r)*w;A[i,adj]=-l*w;A[i,opp]=-r*w
  axes.append(A.tocsc())
 ax,ay=axes;L=sparse.kronsum(ax,ay,format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc');rmse=np.sqrt(np.mean((old._forward(s,L)[0]@W-Y)**2,axis=0));results.append((max(rmse),(dl,dw,ul,uw),rmse))
for result in sorted(results)[:5]:print('fixed stencil',result)
