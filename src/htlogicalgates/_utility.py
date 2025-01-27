ENCODING_FILE = "encodings.json"
CON_FILE = "connectivities.json"
ENCODING_KEY = "enc_t"
CON_KEY = "con"
DESCR_KEY = "desc"


class MissingOptionalLibraryError(Exception):
    pass


def _argument_assignment(options, name, *args, **kwargs):
    class Empty:
        ...

    def try_option(option):
        if len(args) + len(kwargs) != len(option):
            return None
        e = Empty()
        res = dict.fromkeys(option.keys(), e)
        for key, val in zip(res.keys(), args):
            if not isinstance(val, option[key]):
                return None
            else:
                res[key] = val
        for key, val in kwargs.items():
            if not key in res.keys():
                return None
            if res[key] != e:
                raise TypeError(
                    f"{name} got multiple values for argument '{key}'")
            if not isinstance(val, option[key]):
                raise TypeError(f"{name} argument '{key}' is of type '{str(type(val))}', must be of type '{str(option[key])}'")
            res[key] = val
        if not any([i is e for i in res.values()]):
            return res
        return None

    for i, option in enumerate(options):
        result = try_option(option)
        if result != None:
            return i, result
    raise TypeError(f"{name} got invalid arguments")
