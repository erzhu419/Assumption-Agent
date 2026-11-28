from metric_probe import metric_gp
import batch_solver as m
import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; c,b,v,w=m.coordinates(s,x); cv=np.c_[c,v]; bw=np.c_[b,w]; ix=np.arange(len(x))
holds=[('public',ix[52:]),('first',ix[:20]),('middle',ix[24:44]),('corners',np.argsort(np.linalg.norm(x,axis=1))[-20:])]+[(f'x{j}-{side}',np.argsort(x[:,j])[part]) for j in range(4) for side,part in [('low',slice(None,20)),('high',slice(-20,None))]]
for name,h in holds:
 t=np.setdiff1d(ix,h); f=np.linalg.svd(z[t],full_matrices=False)[2][0]; a=z[t]@f
 y0=m.gp(cv[t],cv[h],y[t],True); y1=metric_gp(x[t],x[h],y[t],(0,0,1)); y2=m.gp(b[t],b[h],y[t]); y3=metric_gp(cv[t],cv[h],y[t],(1,1,0));
 a0=m.gp(bw[t],bw[h],a); a1=metric_gp(x[t],x[h],a,(1,1,0)); a2=metric_gp(cv[t],cv[h],a,(1,1,0)); a3=metric_gp(b[t],b[h],a,(1,1,0));
 Y=[y0,y1,y2,y3,.65*y0+.35*y1,.55*y0+.25*y1+.2*y2,.5*y0+.25*y1+.25*y3]; A=[a0,a1,a2,a3,.25*a0+.25*a1+.5*a2,.2*(a0+a1+a3)+.4*a2,(a1+a2+a3)/3];
 print(name,'Y',np.round([np.mean(abs(p-y[h])) for p in Y],5),'Z',np.round([np.mean(abs(p[:,None]*f-z[h])) for p in A],5),'Z horizon',np.round([np.mean(abs(p[:,None]*f-z[h]),0).max() for p in A],5),flush=True)
