#!/usr/bin/env python3
"""Run a command, log wall time, max child RSS and min system MemAvailable."""
import os, resource, subprocess, sys, threading, time
def avail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return int(l.split()[1])
tot = int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1])
mn = [avail()]; stop = False
def poll():
    while not stop:
        mn[0] = min(mn[0], avail()); time.sleep(0.5)
t = threading.Thread(target=poll, daemon=True); t.start()
t0 = time.time()
rc = subprocess.call(sys.argv[2:], cwd=sys.argv[1])
wall = time.time() - t0; stop = True
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"TIMED exit={rc} wall_s={wall:.0f} max_child_rss_gb={ru.ru_maxrss/1048576:.2f} peak_sys_used_gb={(tot-mn[0])/1048576:.2f}")
sys.exit(rc)
