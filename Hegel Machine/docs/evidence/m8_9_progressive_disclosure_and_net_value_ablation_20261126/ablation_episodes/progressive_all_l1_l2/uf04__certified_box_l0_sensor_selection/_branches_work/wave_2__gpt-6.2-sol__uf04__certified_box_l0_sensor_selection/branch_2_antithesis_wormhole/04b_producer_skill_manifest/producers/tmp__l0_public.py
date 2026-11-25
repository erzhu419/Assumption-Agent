import sys, time, pickle
sys.path.insert(0,'/holdout_task')
from certified_l0_solver import solve_certified_box_l0
with open('/holdout_task/public_box_l0_instances.pkl','rb') as f:
    cases=pickle.load(f)
for i, case in enumerate(cases):
    now=time.monotonic(); r=solve_certified_box_l0(case)
    print(i, r['objective'], r['objective']-r['certified_lower_bound'], r['nodes_explored'], time.monotonic()-now)
