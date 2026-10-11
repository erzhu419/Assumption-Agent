"""Summarize a paid offline CPU profile; profiler timings are diagnostic only."""
import json
from pathlib import Path
import pstats

base = Path(__file__).resolve().parent
profile = base / 'profile_root08' / 'full_chain.prof'
stats = pstats.Stats(str(profile))

def label(key):
    filename, line, name = key
    marker = '/hegel_machine/'
    filename = 'hegel_machine/' + filename.split(marker, 1)[1] if marker in filename else filename
    return f'{filename}:{line}:{name}'

def entry(key):
    primitive, calls, own, cumulative, callers = stats.stats[key]
    return {'function': label(key), 'calls': calls, 'primitive_calls': primitive,
            'self_seconds': own, 'cumulative_seconds': cumulative}

functions = [key for key in stats.stats if '/hegel_machine/' in key[0]]
by_self = sorted(functions, key=lambda key: stats.stats[key][2], reverse=True)
by_cumulative = sorted(functions, key=lambda key: stats.stats[key][3], reverse=True)
callers = []
for key in by_self[:8]:
    edges = stats.stats[key][4]
    callers.append({'function': label(key), 'callers': [
        {'function': label(caller), 'statistics': values}
        for caller, values in sorted(edges.items(), key=lambda edge: edge[1][-1], reverse=True)[:12]]})
result = {'scope': 'Complete development root08 six-stage cumulative learning replay. cProfile instrumentation inflates CPU timings; use unprofiled study CPU for paired cost endpoints.',
          'total_seconds': stats.total_tt, 'total_calls': stats.total_calls,
          'top_self': [entry(key) for key in by_self[:25]],
          'top_cumulative': [entry(key) for key in by_cumulative[:25]], 'hotspot_callers': callers}
(profile.parent / 'cpu_profile_summary.json').write_text(json.dumps(result, indent=2) + '\n')
for category in ('top_self', 'top_cumulative'):
    print(category)
    for row in result[category][:12]:
        print(row)
