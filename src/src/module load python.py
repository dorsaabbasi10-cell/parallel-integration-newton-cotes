module load python

cat > collect_c_mpi_sin.py << 'PY'
import re, os

def parse(path):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    m_res = re.search(r"Result\s*:\s*([0-9eE\+\-\.]+)", txt)
    m_tim = re.search(r"Time\s*:\s*([0-9eE\+\-\.]+)\s*seconds", txt)
    if not (m_res and m_tim):
        raise RuntimeError(f"Could not parse {path}")
    return float(m_res.group(1)), float(m_tim.group(1))

rows = []
for p in (28, 56, 112, 224):
    path = f"logs/c_mpi_sin_{p}.txt"
    if not os.path.exists(path):
        continue
    res, t = parse(path)
    nodes = (p + 27) // 28
    rows.append((nodes, p, res, t, path))

rows.sort(key=lambda x: x[1])

print("nodes,processes,result,time_s,file")
for nodes, p, res, t, path in rows:
    print(f"{nodes},{p},{res:.15f},{t:.9f},{path}")
PY

python3 collect_c_mpi_sin.py > c_mpi_sin_results.csv
cat c_mpi_sin_results.csv
