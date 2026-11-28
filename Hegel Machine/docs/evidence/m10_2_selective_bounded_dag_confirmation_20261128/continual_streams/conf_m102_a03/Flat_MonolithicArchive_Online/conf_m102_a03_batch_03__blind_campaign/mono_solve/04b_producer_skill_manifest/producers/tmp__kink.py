exec(open('/tmp/warp_probe.py').read().split("print('B bounds'")[0])
q=np.abs(u); print('q quantiles',np.quantile(q,np.arange(11)/10)); print('test z max',np.max(abs(m._decode(s['test']['x'],m._calibration(s['panel_a'])))))
forms=[('threshold',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.35,1.]),('threshold_lin', lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]+p[2]*x,[.35,1.,.02]),('threshold_kink',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)+p[1]*x,[.35,.02]),('dead_cubic', lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]+p[2]*x**3,[.25,1.7,.02]),('pw3',lambda x,p:np.sign(x)*(np.maximum(abs(x)-p[0],0)+p[2]*np.maximum(abs(x)-p[1],0))+p[3]*x,[.25,.7,-.1,.02]),('power',lambda x,p:np.sign(x)*abs(x)**p[0]+p[1]*x,[2.5,.02]),('clip',lambda x,p:x+p[1]*np.clip(x,-p[0],p[0]),[.45,-.98])]
for name,fn,init in forms:
 def r(p):
  d=aff(fn(u,p));return (d@fit(d,v)-v).ravel()
 p=least_squares(r,init,max_nfev=300).x; measure((name,p),lambda x:fn(x,p))
