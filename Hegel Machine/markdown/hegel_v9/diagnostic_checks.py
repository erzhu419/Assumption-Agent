"""Independent mathematical controls, not Hegel repository tests."""
from pathlib import Path
import json
import numpy as np
x = np.array([-2., -1., 0., 1., 2.])
phi = np.column_stack([x, x*x])
source = phi @ np.array([1.,1.])
target = phi @ np.array([1.,-1.])
full = np.column_stack([np.ones(len(x)),phi])
compressed = np.column_stack([np.ones(len(x)),source])
bf = np.linalg.lstsq(full,target,rcond=None)[0]
bc = np.linalg.lstsq(compressed,target,rcond=None)[0]
result = {
    "scope":"Mathematical illustrations only; no project experiment or test rerun.",
    "source_prediction_identical":bool(np.allclose(source,compressed[:,1])),
    "family_full_rank":int(np.linalg.matrix_rank(full)),
    "family_compressed_rank":int(np.linalg.matrix_rank(compressed)),
    "target_full_mse":float(np.mean((full@bf-target)**2)),
    "target_compressed_mse":float(np.mean((compressed@bc-target)**2)),
    "partial_feasible_found_sets":[[1],[2]],
    "full_feasible_sets":[[1,3],[2,3]],
    "actual_intersection":[3],
    "iid_miss_p_0_01_n48":.99**48,
    "iid_upper_p_zero_of_48_95pct":1-.05**(1/48),
    "receipt_new_compute":12+68724,
    "receipt_target_remaining":270010-153660,
    "receipt_source_remaining":270008-269243,
    "work_improvement_rounded":(0.0073123639-0.0067115732)/0.0073123639,
    "future_degradation_rounded":(0.0164085301-0.0116698350)/0.0116698350,
}
assert result["target_full_mse"] < 1e-20
assert result["target_compressed_mse"] > 1
assert result["receipt_target_remaining"] == 116350
out=Path(__file__).with_name("diagnostic_checks_reproduced.json")
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
