exp_root = "/Users/daniellengyel/flat_sharp/flat_sharp/experiments/MNIST/{}"

a_exp_folder = exp_root.format("Apr14_19-07-42_Daniels-MacBook-Pro-4.local")
a_runs = get_runs(a_exp_folder, ["Loss"])
a_configs = get_configs(a_exp_folder)

b_exp_folder = exp_root.format("Apr14_19-07-45_Daniels-MacBook-Pro-4.local")

b_runs = get_runs(b_exp_folder, ["Loss"])
b_configs = get_configs(b_exp_folder)

b_config_map = get_config_to_id_map(b_configs)
for i in a_runs.keys():
    b_id = get_ids(b_config_map, a_configs[i])[0]
    a_idx = get_sample_idxs(os.path.join(a_exp_folder, "resampling", i))
    b_idx = get_sample_idxs(os.path.join(b_exp_folder, "resampling", b_id))

    for s in a_runs[i].keys():
        for nn in range(100):
            assert a_idx[str(s)][nn] == b_idx[str(s)][nn]
            assert a_runs[i][s]["Loss"]["train"]["net"][str(nn)] ==  b_runs[b_id][s]["Loss"]["train"]["net"][str(nn)]

# test unbiased estimator
r = lambda: np.random.normal(loc=3, size=5000)

Ys = []
N = 200
krr = 0
for i in range(100):
    temp_ys = np.array([r() for _ in range(N)])
    krr += np.sum(temp_ys, axis=0)

    sum_tmp_ys = np.sum(temp_ys, axis=0)
    sum_squared = np.sum(i.dot(i) for i in temp_ys)
    Ys.append((sum_tmp_ys.dot(sum_tmp_ys) - sum_squared) * (1. / (N * (N - 1))))

print(np.mean(Ys))

m = m = [exp_dict["models"]["1592258035.9073222"][str(i)] for i in range(20)] # np.random.randint(0, 100, size=100)
idxs =  np.random.choice(list(range(len(m))), len(m))
a = time.time()
m_right = [copy.deepcopy(m[i]) for i in idxs]
b = time.time()
sample_nets(m, idxs)
c = time.time()
assert all([same_model(m[i], m_right[i]) for i in range(len(m))])

print(idxs)

print(b - a)
print(c - b)

