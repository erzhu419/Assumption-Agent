from alpha_probe import *
from scipy.stats import median_abs_deviation
out=np.empty_like(y)
for lab in np.unique(labs):
 mask=labs==lab
 _,_,res=fit(complete&~mask,'base')
 out[mask]=res[mask]
print('fold residual curve MAD by horizon',median_abs_deviation(out[complete,:-1],axis=0,scale='normal').round(5))
print('curve outliers above .15 by horizon',(abs(out[:,:-1])>.15).sum(0),'rows affected',np.unique((abs(out[:,:-1])>.15).sum(1),return_counts=True))
for name,mask in [('complete',complete),('missing',~complete)]:
 r=out[mask]; cc=r[:,:-1]; rate=r[:,-1]; slope=np.median(cc[:,4:]/t[4:],axis=1)
 print(name,'rate and late curve slope MAD',median_abs_deviation(rate,scale='normal'),median_abs_deviation(slope,scale='normal'),'correlation within central range',np.corrcoef(np.clip(rate,-.08,.08),np.clip(slope,-.08,.08))[0,1])
 for a in [0,.10,.20,.25,.35,.5,.75,.9,.95,1]:
  e=cc-a*rate[:,None]*t
  perlab=[np.mean(np.minimum(abs(e[labs[mask]==l]),.10)) for l in range(4)]
  end=abs(e[:,-1]); sec=end/np.maximum(abs(hold[mask]),.5)
  print(a,'curve capped MAE / worst-lab',round(np.minimum(abs(e),.10).mean(),5),round(max(perlab),5),'end median / MAD slope / secant median',np.round([np.median(end),median_abs_deviation(np.median(e[:,4:]/t[4:],axis=1),scale='normal'),np.median(sec)],5))
