from typing import Callable, Generic, ParamSpec, TypeVar, cast

from chimerax.atomic import Structure, Structures, all_structures
from chimerax.atomic.args import StructuresArg
from chimerax.core.commands import BoolArg, CmdDesc
from chimerax.core.errors import UserError
from chimerax.core.session import Session

from .model import CarbVisModel
from .paperchain import PaperChainModel
from .twister import TwisterModel

P = ParamSpec("P")
R = TypeVar("R")


class CmdFunc(Generic[P, R]):
    desc: CmdDesc

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


def cmd(
    required=(),
    optional=(),
    keyword=(),
    postconditions=(),
    required_arguments=(),
    non_keyword=(),
    hidden=(),
    url=None,
    synopsis=None,
    self_logging=False,
):
    def decorator(func: Callable[P, R]) -> CmdFunc[P, R]:
        ret = cast(CmdFunc, func)
        ret.desc = CmdDesc(
            required,
            optional,
            keyword,
            postconditions,
            required_arguments,
            non_keyword,
            hidden,
            url,
            synopsis,
            self_logging,
        )
        return ret

    return decorator


@cmd(
    optional=[("structures", StructuresArg)],
    keyword=[("replace", BoolArg), ("update", BoolArg)],
)
def paperchain(
    session: Session,
    structures: Structures | None = None,
    replace=True,
    update=True,
):
    """PaperChain description"""

    structures = check_structures(structures, session)

    models: list[PaperChainModel] = []

    for structure in structures:
        structure: Structure

        model = None

        new = True
        if replace:
            model = find_model(PaperChainModel, structure, session)
        if model is None:
            model = PaperChainModel(session, structure, update=update)
            new = True
        else:
            # TODO: update
            new = False

        model.calculate_graphics()

        if new:
            # Add new models to open models list.
            session.models.add([model], parent=model.structure)

        # Make sure replaced surfaces are displayed.
        model.display = True

        models.append(model)

    return models


@cmd(
    optional=[("structures", StructuresArg)],
    keyword=[("replace", BoolArg), ("update", BoolArg)],
)
def twister(
    session: Session,
    structures: Structures | None = None,
    replace=True,
    update=True,
):
    """Twister description"""

    structures = check_structures(structures, session)

    models: list[TwisterModel] = []

    for structure in structures:
        structure: Structure

        model = None

        new = True
        if replace:
            model = find_model(TwisterModel, structure, session)
        if model is None:
            model = TwisterModel(session, structure, update=update)
            new = True
        else:
            # TODO: update
            new = False

        model.calculate_graphics()

        if new:
            # Add new models to open models list.
            session.models.add([model], parent=model.structure)

        # Make sure replaced surfaces are displayed.
        model.display = True

        models.append(model)

    return models


def check_structures(structures: Structures | None, session: Session) -> Structures:
    if structures is None:
        structures = all_structures(session)
        if len(structures) == 0:
            raise UserError("No structures open")
        setattr(structures, "spec", "all structures")
    elif len(structures) == 0:
        msg = "No structures specified"
        if hasattr(structures, "spec"):
            msg += f" by {getattr(structures, 'spec')}"
        raise UserError(msg)
    return structures


ModelT = TypeVar("ModelT", bound=CarbVisModel)


def find_model(
    cls: type[ModelT],
    structure: Structure,
    session: Session,
) -> ModelT | None:
    """Try to find an existing model for the structure"""

    for model in session.models.list(type=cls):
        model: ModelT
        if model.structure == structure:
            return model
    return None
