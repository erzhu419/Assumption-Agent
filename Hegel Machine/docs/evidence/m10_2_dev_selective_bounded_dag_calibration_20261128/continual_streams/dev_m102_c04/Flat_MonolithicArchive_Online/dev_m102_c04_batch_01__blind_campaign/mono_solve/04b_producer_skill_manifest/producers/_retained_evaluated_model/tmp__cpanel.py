from bases import *
from itertools import product
u=s['panel_c']['u']; q=s['panel_c']['v1']; h=s['panel_c']['v2']; W=ls(poly(u,2),q)
def Q(z):return poly(z,2)@W
print('C1 residual',np.std(Q(u)-q),'coefficients',W);print('C2 and C1 samples',np.c_[u,q,h][:14]);print('C2 square coeffs',ls(poly(u,2),h*h));
for name,A in {'square':poly(u,2),'abs':np.c_[poly(u,1),abs(u)],'q':poly(q,1),'absq':np.c_[poly(q,1),abs(q)],'absB':np.c_[poly(u,1),abs(poly(u,1)@cb)],'absBinv':np.c_[poly(u,1),abs((u-cb[0])@np.linalg.inv(cb[1:]))]}.items():
 print('C2',name,'RMSE',*[np.std(A@ls(A,b)-b) for b in [h,h*h]])
for zname,z in coords(u).items():
 print('C1 coord',zname,'linear',np.linalg.norm(ls(poly(z,2),q)[1:5]),'square diagonal ratio',np.linalg.norm(ls(poly(z,2),q)[[5,9,12,14]])/np.linalg.norm(W[5:]))
rv=np.c_[y,t]; ds=coords(x); out=[]
for k,z in ds.items():
 a=Q(z)
 for k1,w in [('same',z),('raw',x),('inv',ds['inv']),('invB',ds['invB']),('a',a)]:
  fs={'lin':np.c_[poly(w,1),a], 'sq':np.c_[poly(w,2),a], 'bil':np.c_[poly(w,1),a, np.einsum('ni,nj->nij',w,a).reshape(-1,16)], 'pair':np.c_[poly(w,1),a,w*a,w/(1+.22*a*a),a/(1+.22*w*w)],'a_rat':np.c_[poly(w,1),a,a/(1+.22*a*a),w*a]}
  for label,f in fs.items():
   for i,r in enumerate(scores(f,rv)):out.append((k,k1,label,i,r))
for col in [1,3]:print('C1 CV',col,*sorted(out,key=lambda v:v[-1][col])[:10],sep='\n')
