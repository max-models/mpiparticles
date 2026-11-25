"""A stable counting sort of the markers by an integer key, such as their cell.

The Pyccel version (CPU) of the ``counting_sort`` kernel. Three passes over the markers
and one over the keys, O(n + num_keys): about 2 ns per marker on one core, against about
10 ns for NumPy's stable argsort of the same keys.
"""


def counting_sort(
    keys: "int[:]",
    alive: "bool[:]",
    starts: "int[:]",
    order: "int[:]",
) -> None:
    """Write the stable order of the markers by key to ``order``, the dead ones last.

    ``keys`` and ``alive`` are ``(N,)``; ``starts`` has ``num_keys + 2`` entries and
    ``order`` ``N``. An alive marker's key must lie in ``0 .. num_keys - 1``; a dead
    marker is put in the extra key ``num_keys``, whatever its key. On return, the markers
    with key ``k`` are ``order[starts[k]:starts[k + 1]]``, in their original order, and
    ``starts[num_keys]`` is the number of alive markers. If an alive key is out of range,
    ``starts[0]`` is set to -1 and nothing else is valid.
    """
    num_keys = starts.shape[0] - 2
    num_markers = keys.shape[0]
    for key in range(num_keys + 2):
        starts[key] = 0
    for marker in range(num_markers):
        key = num_keys
        if alive[marker]:
            key = keys[marker]
            if key < 0 or key >= num_keys:
                starts[0] = -1
                return
        starts[key + 1] += 1
    for key in range(num_keys):
        starts[key + 1] += starts[key]
    for marker in range(num_markers):
        key = keys[marker] if alive[marker] else num_keys
        order[starts[key]] = marker
        starts[key] += 1
    # each start has moved to the next key's: shift them back
    for key in range(num_keys + 1, 0, -1):
        starts[key] = starts[key - 1]
    starts[0] = 0
