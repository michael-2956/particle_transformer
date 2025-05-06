import numpy as np
from pathlib import Path

ks = (
    [("default", None)] +
    list(map(lambda x: ("nl", x), list("1 2 3 4 6 8 12 16 20 24 30 36 42 50".split(" ")))) +
    list(map(lambda x: ("nlcm", x), list("0.1 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1".split(" ")))) +
    list(map(lambda x: ("esm", x), list("0.03125 0.0625 0.125 0.25 0.5 2 4".split(" ")))) +
    list(map(lambda x: ("pesm", x), list("0.03125 0.0625 0.125 0.25 0.5 2 4".split(" "))))
)

a = np.concatenate([
    [1],  # default weight is 1
    np.array(list(map(float, list("1 2 3 4 6 8 12 16 20 24 30 36 42 50".split(" "))))) / 10,
    np.ones(9, dtype=np.float64),
    np.array(list(map(float, list("0.03125 0.0625 0.125 0.25 0.5 2 4".split(" "))))),
    np.array(list(map(float, list("0.03125 0.0625 0.125 0.25 0.5 2 4".split(" "))))),
])

num_splits = 10
groups = [[] for _ in range(num_splits)]
group_weights = np.zeros(num_splits, dtype=np.float64)
pairs = list(zip(ks, a))
pairs.sort(key=lambda x: x[1], reverse=True)
for key, weight in pairs:
    idx = np.argmin(group_weights)
    groups[idx].append(key)
    group_weights[idx] += weight

weight_map = dict(zip(ks, a))
split_grp_list = []
for i, (grp, total_w) in enumerate(zip(groups, group_weights), start=1):
    sorted_grp = sorted(grp, key=lambda k: weight_map[k])
    print(f"Split {i:2d} (total weight {total_w:.4f}): {sorted_grp}")
    split_grp_list.append(sorted_grp)

for i, grp in enumerate(split_grp_list):
    sh_p = Path(f"scaling_experiments/setting_collections/setting_{i+1}.sh")
    sh_p.touch(mode=0o755, exist_ok=True)
    with open(sh_p, "w") as out:
        out.write("#!/bin/bash\n\n")
        for ctype, cval in grp:
            if ctype == "default":
                out.write("# train default model\n")
                out.write("./scaling_experiments/train_and_test_with_settings.sh ParT-default-soap kin\n\n")
            elif ctype == "nl":
                out.write(f"./scaling_experiments/train_and_test_with_settings.sh ParT-nl{cval} kin --total-num-layers {cval}\n\n")
            elif ctype == "nlcm":
                out.write(f"./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm{cval} kin --num-cls-layers-mult {cval}\n\n")
            elif ctype == "esm":
                out.write(f"./scaling_experiments/train_and_test_with_settings.sh ParT-esm{cval} kin --embedding-scale-mult {cval}\n\n")
            elif ctype == "pesm":
                out.write(f"./scaling_experiments/train_and_test_with_settings.sh ParT-pesm{cval} kin --pair-embedding-scale-mult {cval}\n\n")
