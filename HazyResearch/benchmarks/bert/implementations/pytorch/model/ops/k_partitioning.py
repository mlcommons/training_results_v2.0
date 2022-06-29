import torch

from einops import rearrange, repeat


def k_partition(seqlens, ngpus, compare=False):
    """We assume that seqlens are sorted from largest to smallest
    Arguments:
        seqlens: (s,)
        ngpus: int
    Return:
        indices: (s,), each has value from 0 to ngpus - 1.
    """
    device = seqlens.device
    seq_per_gpu = seqlens.shape[0] // ngpus
    assert seqlens.shape[0] == seq_per_gpu * ngpus
    # Folding heuristic from https://link.springer.com/article/10.1007/BF01193837
    arange_idx = torch.arange(ngpus, device=device)
    folding_indices = repeat(torch.cat([arange_idx, arange_idx.flip(-1)]), 'n -> (s n)',
                             s = seq_per_gpu // 2)
    if seq_per_gpu % 2 == 1:
        folding_indices = torch.cat([folding_indices, arange_idx])

    # folding is around 2-4% better than naive, and within 0.5% of ideal.
    # random does about as well as folding, but it's more complicated and requires all GPUs
    # to have the same random states.
    # So I'm just going to use folding for simplicity.
    if compare:
        naive_indices = repeat(arange_idx, 'n -> (s n)', s = seq_per_gpu)

        # torch.randperm doesn't support batching for now
        # https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
        repeats = 100
        random_indices = rearrange(torch.argsort(torch.rand(repeats, seq_per_gpu, ngpus, device=device),
                                                dim=-1),
                                   'b s n -> b (s n)')
        # torch.bincount doesn't support batching
        # https://discuss.pytorch.org/t/batched-bincount/72819
        random_lens = torch.zeros(repeats, ngpus, device=device, dtype=torch.long)
        random_lens.scatter_add_(-1, random_indices, repeat(seqlens, 's -> b s', b=repeats))
        best_random_indices = random_indices[random_lens.amax(dim=-1).argmin()]

        ideal = seqlens.sum() / ngpus
        naive_max = naive_indices.bincount(seqlens).max()
        folding_max = folding_indices.bincount(seqlens).max()
        random_max = best_random_indices.bincount(seqlens).max()
        print(f'Ideal: {seqlens.sum() / ngpus}')
        print(f'Naive: {naive_max}, approx_ratio = {naive_max / ideal:.6}x')
        print(f'Folding: {folding_max}, approx ratio = {folding_max / ideal:.6}x, improve on naive = {naive_max / folding_max:.6}x')
        print(f'Random: {random_max}, approx ratio = {random_max / ideal:.6}x, improve on naive = {naive_max / random_max:.6}x')

    return folding_indices
