import time
from typing import Callable, Iterable, Optional, Sequence, TypeVar, Union
from rich.progress import track


ProgressType = TypeVar("ProgressType")
StyleType = Union[str, "Style"]


def pbar(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
) -> Iterable[ProgressType]:
    return track(
        sequence=sequence,
        description=description,
        total=total,
        auto_refresh=auto_refresh,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second,
        style=style,
        complete_style=complete_style,
        finished_style=finished_style,
        pulse_style=pulse_style,
        update_period=update_period,
        disable=disable,
        show_speed=show_speed,
    )


def prange(rg: Sequence[int], description: str = "processing..."):
    return pbar(range(rg), description=description)
