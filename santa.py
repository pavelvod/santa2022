import pathlib

from tqdm import tqdm
import plotly.express as px
import numpy as np
import pandas as pd
import itertools
from scipy import signal
import matplotlib.pyplot as plt
import numba as nb
import functools
from joblib import Parallel, delayed
import multiprocessing as mp
import networkx as nx
import joblib


def cartesian_to_array(x, y, shape_):
    m, n = shape_[:2]
    i_ = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i_ < 0 or i_ >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i_, j


def get_position(config):
    return functools.reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))


def compress_path(path):
    if len(path) > 2:
        new_path = []
        max_conf_dist = 1
        r = [[] for _ in range(len(path[0]))]
        for p in path:
            for i, c in enumerate(p):
                if len(r[i]) == 0 or r[i][-1] != c:
                    if c not in r[i]:
                        r[i].append(c)
                    else:
                        r[i] = r[i][:r[i].index(c) + 1]
                    assert r[i][-1] == c

        max_conf_dist = max([len(r_) for r_ in r])
        for i in range(max_conf_dist):
            new_conf = []
            for _, r_ in enumerate(r):

                if i < len(r_):
                    c_ = r_[i]
                else:
                    c_ = r_[-1]
                new_conf.append(c_)
            new_path.append(new_conf)
        return new_path
    return path


def rotate_link(vector, direction):
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif y > x and y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif y >= x and y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return (x, y)


def rotate(config, i, direction):
    config = list(config).copy()
    config[i] = rotate_link(config[i], direction)
    return tuple(config)


def get_direction(u, v):
    """Returns the sign of the angle from u to v."""
    direction = np.sign(np.cross(u, v))
    if direction == 0 and np.dot(u, v) < 0:
        direction = 1
    return direction


# @nb.jit(target_backend='cuda')
def reconfiguration_cost(from_config, to_config):
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    assert diffs.max() <= 1
    return np.sqrt(diffs.sum())


# @nb.jit(target_backend='cuda')
def color_cost(from_position, to_position, image_, color_scale=3.0):
    return np.abs(image_[to_position] - image_[from_position]).sum() * color_scale


# Total cost of one step: the reconfiguration cost plus the color cost
# @nb.jit(target_backend='cuda')
def step_cost(from_config, to_config, image_):
    scfg1 = config_to_string(from_config)
    scfg2 = config_to_string(to_config)
    try:
        return G[scfg1][scfg2]['cost']
    except:
        pass
    pos_from = get_position(from_config)
    pos_to = get_position(to_config)
    from_position = cartesian_to_array(pos_from[0], pos_from[1], image_.shape)
    to_position = cartesian_to_array(pos_to[0], pos_to[1], image_.shape)
    cost = (reconfiguration_cost(from_config, to_config) + color_cost(from_position, to_position, image_))
    if G.has_edge(scfg1, scfg2):
        G[scfg1][scfg2]['cost'] = cost
    else:
        G.add_edge(scfg1, scfg2, cost=cost)
    return cost


def get_path_to_point(config, point):
    """Find a path of configurations to `point` starting at `config`."""
    path = [config]
    # Rotate each link, starting with the largest, until the point can
    # be reached by the remaining links. The last link must reach the
    # point itself.
    for i in range(len(config)):
        link = config[i]
        base = get_position(config[:i])
        relbase = (point[0] - base[0], point[1] - base[1])
        position = get_position(config[:i + 1])
        relpos = (point[0] - position[0], point[1] - position[1])
        radius = functools.reduce(lambda r, link: r + max(abs(link[0]), abs(link[1])), config[i + 1:], 0)
        # Special case when next-to-last link lands on point.
        if radius == 1 and relpos == (0, 0):
            config = rotate(config, i, 1)
            if get_position(config) == point:  # Thanks @pgeiger
                path.append(config)
                break
            else:
                continue
        while np.max(np.abs(relpos)) > radius:
            direction = get_direction(link, relbase)
            config = rotate(config, i, direction)
            path.append(config)
            link = config[i]
            base = get_position(config[:i])
            relbase = (point[0] - base[0], point[1] - base[1])
            position = get_position(config[:i + 1])
            relpos = (point[0] - position[0], point[1] - position[1])
            radius = functools.reduce(lambda r, link: r + max(abs(link[0]), abs(link[1])), config[i + 1:], 0)
    assert get_position(path[-1]) == point

    path = compress_path(path)

    return path


def get_path_to_configuration(from_config, to_config):
    path = [from_config]
    config = from_config.copy()
    while config != to_config:
        for i in range(len(config)):
            config = rotate(config, i, get_direction(config[i], to_config[i]))
        path.append(config)
    assert path[-1] == to_config

    path = compress_path(path)

    return path


# @functools.lru_cache(maxsize=None)
def config_to_string(config):
    return ';'.join([' '.join(map(str, vector)) for vector in config])


# Read image as a numpy array:
df_image = pd.read_csv('image.csv')
side = df_image.x.nunique()
radius = df_image.x.max()
image = df_image[['r', 'g', 'b']].values.reshape(side, side, -1)

# Flip X axis and transpose X-Y axes to simplify cartesian to array mapping:
image = image[::-1, :, :]
image = np.transpose(image, (1, 0, 2))

side = image.shape[0]

plt.subplots(figsize=(16, 16))
plt.imshow(image)


def calc_reconfiguration_cost(from_config, to_configs):
    n = to_configs.shape[0]
    diffs = np.abs(from_config.reshape(n_links, 2) - to_configs.reshape(n, n_links, 2)).sum(-1)
    return np.sqrt(diffs.sum(1)).reshape(-1, 1)


def calc_color_cost(from_config, to_configs):
    pos_from = from_config.reshape(n_links, 2).sum(0)
    from_position = cartesian_to_array(pos_from[0], pos_from[1], image.shape)

    out = []
    for to_config in to_configs:
        pos_to = to_config.reshape(n_links, 2).sum(0)
        to_position = cartesian_to_array(pos_to[0], pos_to[1], image.shape)
        out.append(color_cost(from_position, to_position, image))
    return np.array(out).reshape(-1, 1)


def calc_cost(from_config, to_configs):
    reconf_cost = calc_reconfiguration_cost(from_config, to_configs)
    color_cost = calc_color_cost(from_config, to_configs)
    cost = reconf_cost + color_cost
    return cost


n_links = 8
nbr_size = 10


def create_loss_surface(visited):
    _visited = (visited > 0).astype(np.float32)
    h, w = _visited.shape
    h = h + nbr_size * 2
    w = w + nbr_size * 2

    padded_visited = np.ones(shape=(h, w))

    padded_visited[nbr_size:-nbr_size, nbr_size:-nbr_size] = _visited

    a = nbr_size + 1 - np.abs(np.arange(-nbr_size, nbr_size + 1))
    a = a * a.reshape(-1, 1)
    #     a = np.sqrt(a)
    a = a / a.max()

    loss_surface = signal.convolve2d(padded_visited, a, mode='same')[nbr_size:-nbr_size, nbr_size:-nbr_size]

    loss_surface = np.maximum(-999, 1.5 - loss_surface / 100)
    loss_surface = loss_surface * (1 - _visited) + visited * 5
    return loss_surface


def prepare_transition_delta_matrix():
    ilink = []

    for i in [-1, 0, 1]:
        ilink.append((0, i))
        ilink.append((i, 0))
    ilink = list(set(ilink))
    ilink2 = list(itertools.product(ilink, ilink))
    ilink3 = list(itertools.product(ilink2, ilink2))
    ilink4 = list(itertools.product(ilink3, ilink3))
    n = len(ilink4)
    transition_delta = np.zeros(shape=(n, 16), dtype=np.int16)
    for idx, i4 in enumerate(ilink4):
        r = []
        for i3 in i4:
            for i2 in i3:
                for i1 in i2:
                    for i in i1:
                        r.append(i)
        transition_delta[idx, :] = r

    transition_delta = np.unique(transition_delta[:, :n_links * 2], axis=0)
    transition_delta = transition_delta[np.abs(transition_delta).sum(1) > 0]
    return transition_delta


def config_to_arr(config):
    return np.array([list(link) for link in config]).reshape(1, -1).astype(np.int16)


def arr_to_config(arr):
    return [tuple(z) for z in arr.reshape(n_links, 2).tolist()]


def calc_walk_cost(configs):
    cost = 0
    for i in range(1, len(configs)):
        cost += step_cost(configs[i - 1], configs[i], image)
    return cost


offset = 2 ** (n_links - 1)


# @functools.lru_cache(maxsize=None)
def str_to_cfg(cfg):
    out = []
    for lnk in cfg.split(';'):
        x, y = lnk.split(' ')
        x = int(x)
        y = int(y)
        lnk = (x, y)
        out.append(lnk)
    return tuple(out)


def get_neighbors(config, G):
    neighbors = [str_to_cfg(e) for _, e in G.edges(config_to_string(config))]
    if len(neighbors) == 26:
        return neighbors

    nhbrs = (
        functools.reduce(lambda x, y: rotate(x, *y), enumerate(directions), config)
        for directions in itertools.product((-1, 0, 1), repeat=3)
    )
    neighbors = list(filter(lambda c: c != config, nhbrs))
    for cfg in neighbors:
        G.add_edge(config_to_string(config), config_to_string(cfg))
    return neighbors


origin = tuple([(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)])

G = nx.Graph()
visited = np.zeros([side, side])


def draw_path(points, strategy='printing'):
    '''DUMB idea to plot the strategy path to print the card'''
    points_list = list()
    for (i, j) in points:
        points_list.append([i, j])

    data = np.array(points_list)

    # plotting a line plot after changing it's width and height
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(15)

    print(f"Plotting the {strategy} approach")
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


def initial_result():
    result = []
    min_loss = 999
    for cfg1, cfg2 in itertools.combinations(get_neighbors(origin, G), 2):
        try:
            c = np.mean([step_cost(origin, cfg1, image), step_cost(origin, cfg2, image), step_cost(cfg1, cfg2, image)])
            if c < min_loss:
                min_loss = c
                result = [origin, cfg1, cfg2]
        except:
            pass
    for cfg in result:
        _ = get_neighbors(cfg, G)
        x, y = get_position(cfg)
        visited[offset + x, offset + y] += 1

    result.append(origin)

    return result


def get_mutual_neighbors(cfg1, cfg2, G):
    scfg1 = config_to_string(cfg1)
    scfg2 = config_to_string(cfg2)

    mutual_neighbors = G[scfg1][scfg2].get('mutual_neighbors', [])
    if len(mutual_neighbors) == 0:
        G[scfg1][scfg2]['mutual_neighbors'] = list(set(map(config_to_string, get_neighbors(cfg1, G))).intersection(
            set(map(config_to_string, get_neighbors(cfg2, G)))))
        # mutual_neighbors = G[scfg1][scfg2]['mutual_neighbors']

    return list(map(str_to_cfg, G[scfg1][scfg2]['mutual_neighbors']))


def mark_unusable_pairs(result):
    ttl = 0
    for i in range(1, len(result)):
        cfg1 = result[i - 1]
        cfg2 = result[i]

        mutual_neighbors = get_mutual_neighbors(cfg1, cfg2, G)
        cnt = 0
        for cfg in mutual_neighbors:
            x, y = get_position(cfg)
            if visited[offset + x, offset + y] == 0:
                cnt += 1
        if cnt == 0:
            scfg1 = config_to_string(cfg1)
            scfg2 = config_to_string(cfg2)
            G[scfg1][scfg2]['no_usable_friends'] = True
            ttl += 1
    return True


def get_best_from_single_pair(i, cfg1, cfg2, visited, G, image):
    min_c = 999
    mutual_neighbors = get_mutual_neighbors(cfg1, cfg2, G)
    cfg_out = None
    offset = 128
    for cfg in mutual_neighbors:
        x, y = get_position(cfg)
        if visited[offset + x, offset + y] > 0:
            continue
        try:
            c = np.mean([step_cost(cfg, cfg1, image), step_cost(cfg, cfg2, image) - step_cost(cfg1, cfg2, image)])
            if c < min_c:
                min_c = c
                cfg_out = cfg
        except Exception as e:
            print(e)
    return i, cfg_out, min_c


@delayed
def calc_list_of_pairs(lst, visited, G, image):
    res = [get_best_from_single_pair(i, cfg1, cfg2, visited, G, image) for i, cfg1, cfg2 in lst]
    p = np.argmin([r[-1] for r in res])
    i, cfg_out, min_c = res[p]
    return i, cfg_out, min_c


def is_drop_pair(i, result):
    try:
        cfg1 = result[i - 1]
        cfg2 = result[i]
        scfg1 = config_to_string(cfg1)
        scfg2 = config_to_string(cfg2)
        return ~G[scfg1][scfg2].get('no_usable_friends', False), i, cfg1, cfg2
    except:
        return False, i, cfg1, cfg2
    # return True, i, cfg1, cfg2


n_jobs = mp.cpu_count() - 1

result = []
for p in pathlib.Path('.').glob('submission_*.csv'):
    r = [str_to_cfg(a) for a in pd.read_csv(p).configuration.tolist()]
    if len(r) > len(result):
        result = r
        G = joblib.load(f'G_{len(result)}.G')
        visited = np.load(f'visited_{len(result)}.npy')

if len(result) == 0:
    result = initial_result()

for cfg in result:
    get_neighbors(cfg, G)

print(f'starting with list of {len(result)} points')

total = side * side - 1
pbar = tqdm(total=total)

pbar.update(len(result))
for iteration in range(total):
    lst = []
    for i in range(1, len(result)):
        isOK, i, cfg1, cfg2 = is_drop_pair(i, result)
        if isOK:
            lst.append((i, cfg1, cfg2))

    _njobs = n_jobs if len(result) > n_jobs * 10 else 1
    n_in_list = len(lst) // _njobs + 1

    lst1 = []
    for idx in range(_njobs):
        lst1.append(calc_list_of_pairs(lst[idx * n_in_list: (idx + 1) * n_in_list], visited, G, image))

    res = Parallel(n_jobs=_njobs, require='sharedmem')(lst1)

    p = np.argmin([r[-1] for r in res])
    i, cfg, c = res[p]
    x, y = get_position(cfg)
    result = result[:i] + [cfg] + result[i:]
    cost = calc_walk_cost(result)
    visited[offset + x, offset + y] += 1
    pbar.update(1)
    if iteration % 100 == 99:
        mark_unusable_pairs(result)

    if len(result) % 250 == 0:
        submission = pd.Series(
            [config_to_string(config) for config in result],
            name="configuration",
        ).to_csv(f'submission_{len(result)}.csv', index=False)
        joblib.dump(G, f'G_{len(result)}.G')
        # G = joblib.load(f'G_{len(result)}.G')
        np.save(f'visited_{len(result)}.npy', visited)
        visited = np.load(f'visited_{len(result)}.npy')
        print(f'score: {calc_walk_cost(result)}\t average: {calc_walk_cost(result) / len(result)}')

print(f'score: {calc_walk_cost(result)}\t average: {calc_walk_cost(result) / len(result)}')

submission = pd.Series(
    [config_to_string(config) for config in result],
    name="configuration",
).to_csv('submission.csv', index=False)

px.imshow(visited)

test_points = [get_position(i) for i in result]
draw_path(test_points, 'greed-approach')

fig, ax = plt.subplots(1, 3, figsize=(32, 32 * 3))
ax[0].imshow(image, alpha=0.7)
ax[1].imshow(visited.T[::-1, :] > 0, alpha=0.3)

ax[2].imshow(image, alpha=0.7)
ax[2].imshow(visited.T[::-1, :] > 0, alpha=0.3)

plt.show()
