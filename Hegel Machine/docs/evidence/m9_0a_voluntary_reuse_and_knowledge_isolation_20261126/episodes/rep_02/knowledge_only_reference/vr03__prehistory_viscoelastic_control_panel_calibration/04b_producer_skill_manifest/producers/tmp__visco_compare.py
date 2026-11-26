import runpy, contextlib, io
with contextlib.redirect_stdout(io.StringIO()): d=runpy.run_path('/tmp/visco_eval.py')
np=d['np']; m=d['m']; auc=d['roc_auc_score']; data=d['data']
for i,e in enumerate(data):
 c=e['cohort']; g=e['gt']; t=c['train']; q=c['query']; cs=t['center_id']; ts=t['stratum']; ys=t['y']; taus=np.asarray(c['taus']); raw=m._extract(t,taus); z=m._align(raw,t['controls'],cs); rq=m._extract(q,taus); zq=m._align(rq,q['controls'],q['center_id']);
 for h in [0,1,2,3,'query']:
  tr=cs!=h if h!='query' else np.ones(cs.size,dtype=bool); v=(z[~tr],ts[~tr],ys[~tr],cs[~tr]) if h!='query' else (zq,q['stratum'],g['y_query'],q['center_id']); w=m._cell_weights(cs[tr],ts[tr]); a,b=m._resolve_missing(z[tr],v[0],w); x=m._features(a,ts[tr]); y=ys[tr].astype(float); xv=m._features(b,v[1]); results=[]
  for family in ['baseline','less ridge','complete train','missing marginal','center intercept','raw pooled']:
   ww=w.copy(); xx=x; vv=xv
   if family=='complete train': ww *= np.isfinite(z[tr]).all(1)
   if family=='center intercept':
    indicators=(cs[tr,None]==np.unique(cs[tr])); xx=np.column_stack([x,indicators-1/indicators.shape[1]]); vv=np.column_stack([xv,np.zeros((len(xv),indicators.shape[1]))]);
   if family=='raw pooled':
    ar,br=m._resolve_missing(raw[tr],raw[~tr] if h!='query' else rq,np.ones(tr.sum())); xx=m._features(ar,ts[tr]); vv=m._features(br,v[1]); ww=np.ones(tr.sum())
   coef=m._logistic(xx,y,ww,ridge=.03 if family=='less ridge' else 1/3); score=vv@coef; p=m.expit(score)
   if family=='missing marginal':
    sd=np.nanstd(z[tr],axis=0); var=(~np.isfinite(v[0])) @ (coef[:3]*sd)**2; p=m.expit(score / np.sqrt(1+np.pi/8*var))
   results.append((family,[(round(auc(v[2][j],p[j]),4),round(np.mean((p[j]-v[2][j])**2),4)) for k in np.unique(v[3]) for j in [v[3]==k]]))
  print(i,h,results)
