 import sys
 import pandas as pd
 import matplotlib.pyplot as plt

 path = sys.argv[1]
 df = pd.read_csv(path)

 # 只保留我们关心的列（避免空列干扰）
 cols = ["runtime","rps","p50_us","p95_us","p99_us","dropped","ok","err"]
 df = df[cols]

 # outlier: p99 > 10ms (10000us) 你论文 SLO 可直接用
 df["p99_outlier"] = df["p99_us"] > 10000

 g = df.groupby(["runtime","rps"], as_index=False).agg(
     runs=("p99_us","size"),
     p50_median=("p50_us","median"),
     p95_median=("p95_us","median"),
     p99_median=("p99_us","median"),
     p99_max=("p99_us","max"),
     dropped_sum=("dropped","sum"),
     outlier_cnt=("p99_outlier","sum"),
 )

 g["outlier_rate"] = g["outlier_cnt"] / g["runs"]

 out_csv = path.replace(".csv","_agg.csv")
 g.to_csv(out_csv, index=False)
 print("Wrote:", out_csv)

 # plot median p99
 plt.figure()
 for rt, sub in g.groupby("runtime"):
     plt.plot(sub["rps"], sub["p99_median"], marker="o", label=f"{rt} median p99")
 plt.xlabel("RPS")
 plt.ylabel("p99 median (us)")
 plt.title("HTTP realistic baseline: median p99 vs RPS")
 plt.legend()
 plt.tight_layout()
 png1 = path.replace(".csv","_p99_median.png")
 plt.savefig(png1)
 print("Wrote:", png1)

 # plot max p99 (stability)
 plt.figure()
 for rt, sub in g.groupby("runtime"):
     plt.plot(sub["rps"], sub["p99_max"], marker="o", label=f"{rt} max p99")
 plt.xlabel("RPS")
 plt.ylabel("p99 max (us)")
 plt.title("HTTP realistic baseline: max p99 vs RPS (stability)")
 plt.legend()
 plt.tight_layout()
 png2 = path.replace(".csv","_p99_max.png")
 plt.savefig(png2)
 print("Wrote:", png2)
