exec(open('/tmp/joint.py').read().split("for bgshift")[0])
# evaluate joint fit per compound, report errors
errs=[]
for k in gt:
 cs=[];ys=[];pis=[];gs=[]
 for pi,(w,bg,co) in enumerate(plates):
  for z in w:
   if z['well_type']=='COMPOUND' and z['compound_id']==k:
    cs.append(z['log10_conc_uM']);ys.append(z['fluorescence']-bg);pis.append(pi);gs.append((b([z['row']],[z['col']])@co)[0])
 c=np.array(cs);y=np.array(ys);pi=np.array(pis);g=np.array(gs)
 def pred(q): return g*q[2:][pi]/(1+10**(q[1]*(c-q[0])))
 q0=np.r_[[-.3,1.05],[2200,2500,2800,3100]]
 fit=least_squares(lambda q:y-pred(q),q0,bounds=([-4,.2,*([500]*4)],[3,3,*([6000]*4)]),loss='soft_l1',f_scale=35,max_nfev=3000)
 e=fit.x[:2]-[gt[k]['log10_ic50_uM'],gt[k]['hill_slope']];errs.append(e);print(k,fit.x[:2],e)
print('MAX',np.max(np.abs(errs),axis=0))
