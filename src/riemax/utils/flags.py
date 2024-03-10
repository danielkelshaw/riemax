import pathlib

from absl import flags

flags.disclaim_key_flags()


class _PathParser(flags.ArgumentParser):
    def parse(self, value):
        return pathlib.Path(value)


class _PathSerializer(flags.ArgumentSerializer):
    def serialize(self, value):
        return str(value)


def DEFINE_path(name, default, help, flag_values=flags._flagvalues.FLAGS, **kwargs):  # noqa: N802
    return flags.DEFINE(_PathParser(), name, default, help, flag_values, _PathSerializer(), **kwargs)
