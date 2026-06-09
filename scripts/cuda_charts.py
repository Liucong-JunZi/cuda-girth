#!/usr/bin/env python3
"""CUDA-standard girth speedup charts: speedup vs problem size, multi-run error bars."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, math, signal, time, json, sys
import networkx as nx
from cuda_girth import girth as cu_girth

OUT = '/cugrith/cuda-girth/scripts/plots'
os.makedirs(OUT, exist_ok=True)
N_RUNS = 5  # multi-run for std-dev
NX_TIMEOUT = 15

class TimeoutError(Exception): pass
def _h(s,f): raise TimeoutError()
def nx_g(G):
    if G.number_of_nodes()<3: return float('inf')
    signal.signal(signal.SIGALRM,_h); signal.alarm(NX_TIMEOUT)
    try: c=nx.minimum_cycle_basis(G); signal.alarm(0); return min(len(x) for x in c) if c else float('inf')
    except: signal.alarm(0); return None

def bench(name, G):
    # warmup
    cu_girth(G)
    # NX
    t0=time.perf_counter(); ng=nx_g(G); nxt=(time.perf_counter()-t0)*1000
    # CUDA multi-run
    cts=[]
    for _ in range(N_RUNS):
        t0=time.perf_counter(); cu_girth(G); cts.append((time.perf_counter()-t0)*1000)
    cu_mean=np.mean(cts); cu_std=np.std(cts)
    ok = (ng==cu_girth(G)) if ng is not None else True
    sp = nxt/cu_mean if cu_mean>0 and nxt>0 and ng is not None else 0
    return {'name':name,'n':G.number_of_nodes(),'m':G.number_of_edges(),
            'nx_ms':nxt,'cu_mean':cu_mean,'cu_std':cu_std,'sp':sp,'ok':ok}

# ---- Collect data ----
rr = []
def t(nm, G): rr.append(bench(nm,G)); print(f"  {nm}")

print("Collecting...")
for nm,g in [("Petersen",nx.petersen_graph),("Dodec.",nx.dodecahedral_graph),
    ("Icosa.",nx.icosahedral_graph),("Desargues",nx.desargues_graph),("Tutte",nx.tutte_graph)]:
    t(nm,g())
for a,b in [(5,5),(10,10),(20,20),(30,30)]: t(f"K_{a},{b}",nx.complete_bipartite_graph(a,b))
for r,c in [(5,5),(10,10),(15,15)]: t(f"Grid{r}x{c}",nx.grid_2d_graph(r,c))
for n_ in [50,100,200,500,1000]:
    for p in [0.05,0.1,0.2]: t(f"ER{n_}p{p}",nx.erdos_renyi_graph(n_,p,seed=42))
for n_ in [50,100,200,500]:
    for d in [2,4]:
        if d<n_: t(f"BA{n_}m{d}",nx.barabasi_albert_graph(n_,d,seed=7))
for d in [3,4,5]:
    for n_ in [50,100,200,500]:
        try: t(f"reg{d}-{n_}",nx.random_regular_graph(d,n_,seed=123))
        except: pass
for n_ in [50,100,200]:
    for k in [4,6]:
        if k<n_: t(f"WS{n_}k{k}",nx.watts_strogatz_graph(n_,k,0.2,seed=99))
for n_ in [10,30,100]: t(f"Cycle{n_}",nx.cycle_graph(n_))

# Save
with open(f'{OUT}/bench_data.json','w') as f: json.dump(rr,f,indent=2,default=str)
print(f"\nCollected {len(rr)} points, {sum(1 for r in rr if r['ok'])} correct")

# ---- Family colors ----
def fam(r):
    nm=r['name']
    if nm.startswith('K_'): return 'Bipartite','#e74c3c'
    if nm.startswith('ER'): return 'Erdős–Rényi','#3498db'
    if nm.startswith('BA'): return 'Barabási–Albert','#2ecc71'
    if nm.startswith('reg'): return 'Regular','#9b59b6'
    if nm.startswith('WS'): return 'Watts-Strogatz','#f39c12'
    if nm.startswith('Grid'): return 'Grid','#1abc9c'
    if nm.startswith('Cycle'): return 'Cycle','#95a5a6'
    return 'Named','#34495e'
for r in rr: r['family'],r['color']=fam(r)

plt.rcParams.update({'font.size':11,'axes.titlesize':15,'axes.labelsize':13,'legend.fontsize':9})

# ===== Chart 1: Speedup vs |V| (CUDA standard) =====
fig,ax=plt.subplots(figsize=(12,7))
families_seen=set()
for r in rr:
    if r['sp']<=0: continue
    lbl = r['family'] if r['family'] not in families_seen else ''
    families_seen.add(r['family'])
    ax.errorbar(r['n'], r['sp'], yerr=r['cu_std']/r['cu_mean']*r['sp'] if r['cu_mean']>0 else 0,
                fmt='o', color=r['color'], markersize=8, capsize=3, label=lbl, alpha=0.8)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Break-even (1×)')
ax.set_xlabel('Number of vertices |V|', fontsize=14)
ax.set_ylabel('Speedup (NetworkX / CUDA)', fontsize=14)
ax.set_title('CUDA Girth Speedup vs Graph Size | RTX 3090', fontsize=17, fontweight='bold')
ax.set_xscale('log')
ax.legend(loc='upper left', fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout(); fig.savefig(f'{OUT}/A_speedup_vs_n.png',dpi=150); print(f"✅ A")

# ===== Chart 2: Speedup vs |E| =====
fig,ax=plt.subplots(figsize=(12,7))
families_seen=set()
for r in rr:
    if r['sp']<=0: continue
    lbl = r['family'] if r['family'] not in families_seen else ''
    families_seen.add(r['family'])
    ax.errorbar(r['m'], r['sp'], yerr=r['cu_std']/r['cu_mean']*r['sp'] if r['cu_mean']>0 else 0,
                fmt='s', color=r['color'], markersize=8, capsize=3, label=lbl, alpha=0.8)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Break-even')
ax.set_xlabel('Number of edges |E|', fontsize=14)
ax.set_ylabel('Speedup', fontsize=14)
ax.set_title('CUDA Girth Speedup vs Edge Count | RTX 3090', fontsize=17, fontweight='bold')
ax.set_xscale('log')
ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.2)
plt.tight_layout(); fig.savefig(f'{OUT}/B_speedup_vs_m.png',dpi=150); print(f"✅ B")

# ===== Chart 3: Wall-clock time comparison (side-by-side) =====
ok_r = sorted([r for r in rr if r['ok']], key=lambda r: r['n'])
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,7))
# NX times
nx_t = [r['nx_ms'] for r in ok_r if r['nx_ms'] is not None]
cu_t = [r['cu_mean'] for r in ok_r]
lbls = [r['name'] for r in ok_r]
x = np.arange(len(lbls))
w = 0.35
ax1.bar(x-w/2, [min(t,5000) for t in nx_t], w, color='#e74c3c', label='NetworkX', alpha=0.8)
ax1.bar(x+w/2, cu_t, w, color='#3498db', label='CUDA (RTX 3090)', alpha=0.8)
ax1.set_xticks(x[::max(1,len(x)//20)]); ax1.set_xticklabels([lbls[i] for i in range(0,len(x),max(1,len(x)//20))], rotation=60, fontsize=7)
ax1.set_ylabel('Time (ms)'); ax1.set_title('Wall-Clock Time (capped at 5s)'); ax1.legend()
# CUDA times only (log scale)
ax2.bar(range(len(ok_r)), cu_t, color='#3498db', alpha=0.8)
ax2.set_yscale('log'); ax2.set_ylabel('CUDA time (ms, log)'); ax2.set_title('CUDA Runtime')
ax2.set_xticks(x[::max(1,len(x)//20)]); ax2.set_xticklabels([lbls[i] for i in range(0,len(x),max(1,len(x)//20))], rotation=60, fontsize=7)
fig.suptitle('Girth Computation: NetworkX vs CUDA | RTX 3090', fontsize=16, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/C_wallclock.png',dpi=150); print(f"✅ C")

# ===== Chart 4: GPU utilization-quality (speedup vs average degree) =====
fig,ax=plt.subplots(figsize=(10,7))
for r in rr:
    if r['sp']<=0: continue
    avg_deg = 2*r['m']/r['n'] if r['n']>0 else 0
    ax.errorbar(avg_deg, r['sp'], yerr=r['cu_std']/r['cu_mean']*r['sp'] if r['cu_mean']>0 else 0,
                fmt='D', color=r['color'], markersize=10, capsize=3, alpha=0.8)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Average degree (2|E|/|V|)', fontsize=14)
ax.set_ylabel('Speedup', fontsize=14)
ax.set_title('Speedup vs Graph Density | RTX 3090', fontsize=17, fontweight='bold')
# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=c, label=f) for f,c in sorted(set((r['family'],r['color']) for r in rr), key=lambda x: x[0])]
ax.legend(handles=legend_patches, fontsize=8, ncol=2, loc='lower right')
ax.grid(alpha=0.2)
plt.tight_layout(); fig.savefig(f'{OUT}/D_speedup_vs_degree.png',dpi=150); print(f"✅ D")

# ===== Summary =====
ok_ = [r['sp'] for r in rr if r['sp']>0]
geo = math.exp(sum(math.log(s) for s in ok_)/len(ok_))
med = sorted(ok_)[len(ok_)//2]
print(f"\n{'='*50}")
print(f"Graphs: {len(rr)}  |  Correct: {len(ok_)}  |  Errors: {len(rr)-len(ok_)}")
print(f"Speedup: geomean={geo:.1f}×  median={med:.1f}×  max={max(ok_):.1f}×")
print(f"{'='*50}")
print(f"Charts → {OUT}/")
