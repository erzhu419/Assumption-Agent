import json,ast,re,pathlib
p=pathlib.Path('/holdout_task/node_00.py'); original=pathlib.Path('/tmp/node00_original.py').read_text()
r=json.loads(pathlib.Path('/tmp/surrogate.result').read_text().splitlines()[0]); m=r['cv_metrics']; curr=max(m.values())
s=pathlib.Path('/tmp/grammar.fixed.log').read_text(); rows=re.search(r'^BEST (.+)$',s,re.M)
if rows:
    best=ast.literal_eval(rows[1])[0]
    if best[0]<min(curr,0.246582) and re.fullmatch(r'[zBCD+*/~()\-:Isqmixrat]+',best[1]):
        # Grammar operators are explicitly interpreted, not evaluated as code.
        ops={'*':'mul','/':'div','~':'gate','+':'add','-':'sub','sq':'sq','mix':'mix','rat':'rat'}
        nm,out=best[1].split(':'); pat=r'z[BCD]{0,2}'
        mm=re.fullmatch(r'\(('+pat+r')(sq|mix|rat|[+*/~\-])('+pat+r')\)([*/])('+pat+r')',nm)
        if mm:
            a,op,b,op2,c=mm.groups(); expr=f"apply('{ops[op2]}',apply('{ops[op]}',walk('{a}'),walk('{b}')),walk('{c}')"
        else:
            mm=re.fullmatch(r'('+pat+r')(sq|mix|rat|[+*/~\-])('+pat+r')(?:(\+|-)('+pat+r'))?',nm)
            if mm:
                a,op,b,op2,c=mm.groups(); expr=f"apply('{ops[op]}',walk('{a}'),walk('{b}'))"; expr=f"apply('{ops[op2]}',{expr},walk('{c}'))" if op2 else expr
            else:
                expr=f"walk('{nm}')" if re.fullmatch(pat,nm) else None
        if expr:
            d='''    h=_fit(study['panel_c'],'v1',lambda q,s:np.arctan(s*q),(.1,5.))
    panel=study['panel_c']; du,dv=panel['u'],panel['v2']
    def db(q,s): return np.column_stack((np.ones(len(q)),np.expm1(-s*q*q),q,q*q))
    def dl(s):
        a=db(du,s); inv=np.linalg.pinv(a)
        return np.mean(np.abs((dv-a@(inv@dv))/(1-np.sum(a*inv.T,axis=1))[:,None]))
    ds=minimize_scalar(dl,bounds=(.01,1.),method='bounded').x
    dp=np.linalg.lstsq(db(du,ds),dv,rcond=None)[0]
    maps={'B':b,'C':h,'D':lambda q:db(q,ds)@dp,'I':lambda q:q}
    def walk(name):
        out=z
        for ch in name[1:]: out=maps[ch](out)
        return out
    def apply(op,x,y):
        if op=='mul': return x*y
        if op=='div': return x/(1+y*y)
        if op=='gate': return x*np.tanh(y)
        if op=='add': return x+y
        if op=='sub': return x-y
        if op=='sq': return x+y*y
        if op=='mix': return x*(1+y)
        return x/(1+np.abs(y))
'''
            pathlib.Path('/tmp/alternate_coord.py').write_text(original.replace('    c=d(z)-b(d(z))',d+f"    c=maps['{out}']({expr})"))
if curr>=0.246582:
    p.write_text(original)
