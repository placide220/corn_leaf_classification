**Setup:** 30 simulated users, 2m per run, each user sends a leaf image to `/predict` every 1–2 s (plus occasional `/health`). Each API container is limited to 1 CPU / 2 GB RAM behind an nginx `least_conn` load balancer.

| Containers | Requests | Failures | Failure % | Requests/s | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) | Load split per replica | Throughput vs 1st |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 510 | 0 | 0.0 | 4.3 | 5034 | 5000 | 11000 | 14000 | 14941 | 100% | 1.0× |

![Load test comparison](results/load_test_comparison.png)
