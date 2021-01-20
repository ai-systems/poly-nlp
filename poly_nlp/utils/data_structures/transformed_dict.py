from collections.abc import MutableMapping


class TransformedDict(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
    function before accessing the keys"""

    def __init__(self, combined_query, *args, **kwargs):
        self.combined_query = combined_query
        dict_values = [(id, id_map) for id, id_map in combined_query.items()]
        self.real_store = dict()
        self.real_store.update(dict(*args))
        self.store = dict()
        self.update(dict(dict_values))  # use the free update to set keys

    def get_key(self, id):
        return self.combined_query[id]

    def __getitem__(self, key):
        key, pos = self.get_key(key)
        val_dict = {}
        for k, v in self.real_store[key].items():
            val_dict[k] = v[pos]
        return val_dict

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
