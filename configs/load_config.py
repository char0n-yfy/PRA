import copy
import os
import yaml

from configs.default import get_config as get_default_config


class ConfigNode(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigNode):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, ConfigNode) else x for x in v]
            else:
                out[k] = v
        return out


def _to_node(obj):
    if isinstance(obj, dict):
        node = ConfigNode()
        for k, v in obj.items():
            node[k] = _to_node(v)
        return node
    if isinstance(obj, list):
        return [_to_node(x) for x in obj]
    return obj


def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def get_config(mode_string):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root, f"configs/{mode_string}_config.yml"),
        os.path.join(root, f"configs/base/{mode_string}_config.yml"),
    ]
    for config_file in candidates:
        if os.path.isfile(config_file):
            return get_config_from_file(config_file)
    raise FileNotFoundError(
        f"Config mode `{mode_string}` not found. Tried: {candidates}"
    )


def get_config_from_file(config_file):
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    base = get_default_config()
    _deep_update(base, config_dict)
    return _to_node(base)
