import dataclasses
import dcargs

from typing import Tuple, Union


@dataclasses.dataclass(frozen=True)
class Subtype:
    data: int = 1


@dataclasses.dataclass(frozen=True)
class TypeA:
    subtype: Subtype = Subtype(1)


@dataclasses.dataclass(frozen=True)
class TypeB:
    subtype: Subtype = Subtype(2)


@dataclasses.dataclass(frozen=True)
class Wrapper:
    supertype: Union[TypeA, TypeB] = TypeA()


if __name__ == "__main__":
    wrapper = dcargs.cli(Wrapper)  # errors when running with supertype:type-a
    print(wrapper)
