"""Phase-2 topology-generalization scaling configs. D2{CS,ICOD} / D3{+RLL} / D5{+ICO,+MPAS}, 3 seeds.
Volume-controlled: epochs set so (#train_pairs * epochs) ~= 400 across levels, so only DIVERSITY differs,
not total gradient steps. Shared held-out val (CS<->ICOD r32). Held-out HEALPix + CSRR are NOT trained
on (zero-shot eval done post-hoc). Base only (corrector dropped)."""
import json, copy

base_t = json.load(open("configs/v20b_base_diverse_topologies_l24_a2p0_mink8.json"))

VAL = ["CS-r32_to_ICOD-r32", "ICOD-r32_to_CS-r32"]   # shared, held out of ALL levels (in-dist everywhere)
TEST_MON = "CS-r32_to_HP-n32"                          # zero-shot monitor only (never used for selection)

D2 = ["CS-r16_to_ICOD-r16", "ICOD-r16_to_CS-r16", "CS-r64_to_ICOD-r64", "ICOD-r64_to_CS-r64"]
RLL = ["CS-r32_to_RLL-r90-180", "RLL-r90-180_to_CS-r32", "RLL-r90-180_to_CS-r16", "RLL-r30-60_to_CS-r16",
       "ICOD-r32_to_RLL-r90-180", "RLL-r90-180_to_ICOD-r16", "RLL-r30-60_to_ICOD-r16"]
ICOMPAS = ["CS-r32_to_ICO-r32", "ICO-r32_to_CS-r32", "ICOD-r32_to_ICO-r32", "ICO-r32_to_ICOD-r32",
           "CS-r32_to_MPAS-r4", "MPAS-r4_to_CS-r32", "ICOD-r32_to_MPAS-r4", "MPAS-r4_to_ICOD-r32",
           "RLL-r90-180_to_ICO-r32", "ICO-r32_to_RLL-r90-180", "RLL-r90-180_to_MPAS-r4",
           "MPAS-r4_to_RLL-r90-180", "ICO-r32_to_MPAS-r4", "MPAS-r4_to_ICO-r32"]
LEVELS = {"D2": D2, "D3": D2 + RLL, "D5": D2 + RLL + ICOMPAS}
K = 400  # target total pair-steps (volume control)

made = []
for lv, pairs in LEVELS.items():
    epochs = max(8, round(K / len(pairs)))
    for seed in (0, 1, 2):
        tag = "p2_%s_s%d" % (lv, seed)
        c = copy.deepcopy(base_t)
        c["run_name"] = tag
        c["model_tag"] = "bipartite_gnn_sinkhorn_%s_kdist_a2p0_mink8" % tag
        c["pairs"] = sorted(set(pairs + VAL + [TEST_MON]))
        tr = c["training"]
        tr["train_pairs"] = pairs
        tr["val_pair"] = VAL[0]
        tr["checkpoint_pairs"] = VAL
        tr["test_pair"] = TEST_MON
        tr["epochs"] = epochs
        tr["seed"] = seed
        json.dump(c, open("configs/%s.json" % tag, "w"), indent=2)
        made.append((tag, len(pairs), epochs))

for tag, npairs, ep in made:
    print("wrote configs/%s.json  train_pairs=%d epochs=%d (pair-steps=%d)" % (tag, npairs, ep, npairs * ep))
