exec(open('/tmp/wall_probe.py').read().split('for name,p in')[0])
axes=[]
for z in 'xy':
 k=s['kappa_'+z]; v=s['vel_'+z]; h=s['L'+z]/(len(k)-1); a=m._axis(k,v,h,0.).tolil()
 for b in (0,len(k)-1):
  left=(b-1)%len(k); right=(b+1)%len(k)
  l=(k[b]+k[left])/(2*h*h)+max(v[b],0)/h
  r=(k[b]+k[right])/(2*h*h)+max(-v[b],0)/h
  a[b,left]=-l; a[b,right]=-r; a[b,b]=l+r
  inward,seam=(r,l) if b==0 else (l,r)
  print(z,b,'periodic multipliers',inward/(k[b]/h**2),seam/((k[0]+k[-1])/(2*h*h)))
 axes.append(a.tocsc())
err=m._responses(s,axes)[0]@w-y
print('periodic FVM pulse RMSE',rms(err),'max',abs(err).max())
