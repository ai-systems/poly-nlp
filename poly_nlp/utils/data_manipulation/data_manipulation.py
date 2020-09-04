from itertools import islice


def create_dict_chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def create_list_chunks(data, size):
    size = max(1, size)
    return (data[i : i + size] for i in range(0, len(data), size))
