from typing import Callable, ParamSpec, Protocol, TypeVar, cast

from chimerax.atomic import Atoms, Bonds, all_atoms, all_bonds
from chimerax.atomic.args import AtomsArg, BondsArg
from chimerax.core.commands import (
    BoolArg,
    CmdDesc,
    EmptyArg,
    EnumOf,
    FloatArg,
    IntArg,
    Or,
)
from chimerax.core.errors import UserError
from chimerax.core.session import Session

from .carbs import dihedral_colormap, dihedral_norm_colormap, paperchain_colormap
from .coloring import color_linkage_bonds
from .paperchain import PaperChainModel
from .strand import StrandModel
from .twister import TwisterModel

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class CmdFunc(Protocol[P, R]):
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
    """A decorator to create a CmdFunc from a function."""

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
    optional=[("atoms", AtomsArg)],
    keyword=[
        ("replace", BoolArg),
        ("update", BoolArg),
        ("bipyramid_height", FloatArg),
        ("max_ring_size", IntArg),
        ("tex_formula", EnumOf(("stripes", "grid", "diamond", "rings", "waves"))),
        ("tex_period", IntArg),
        ("tex_duty", FloatArg),
    ],
    synopsis="Adds a PaperChain visualization to structures.",
)
def paperchain(
    session: Session,
    atoms: Atoms | None = None,
    replace=True,
    update=True,
    bipyramid_height=1.0,
    max_ring_size=10,
    tex_formula=None,
    tex_period=128,
    tex_duty=0.5,
):
    """
    Adds a PaperChain visualization to structures.

    Args:
        structures: The structures to add the visualization to.
        replace: Whether to replace existing PaperChain models.
        update: Whether to automatically update the model if the
            structure changes.
        bypyramid_height: The height of the PaperChain polygon.
        max_ring_size: A ring size limit when finding rings.
        tex_formula: The formula to use for the ring texture.
            One of 'stripes', 'grid', 'diamond', 'rings', 'waves'
        tex_period: The length of the texture period, in pixels.
        tex_duty: The fractional duty cycle of the repeating texture.
    """

    atoms = check_atoms(atoms, session)

    if replace:
        all_models = {
            m.atoms.hash(): m for m in session.models.list(type=PaperChainModel)
        }
    else:
        all_models = {}

    models: list[PaperChainModel] = []

    for structure, model_atoms in atoms.by_structure:
        model = all_models.get(model_atoms.hash())
        if model is None:
            name = f"{structure.name} PaperChain"
            model = PaperChainModel(
                session,
                model_atoms,
                name,
                update=update,
                bipyramid_height=bipyramid_height,
                max_ring_size=max_ring_size,
                tex_formula=tex_formula,
                tex_period=tex_period,
                tex_duty=tex_duty,
            )
            new = True
        else:
            model.update_params(
                update=update,
                bipyramid_height=bipyramid_height,
                max_ring_size=max_ring_size,
                tex_formula=tex_formula,
                tex_period=tex_period,
                tex_duty=tex_duty,
            )
            new = False

        model.update()

        if new:
            if replace:
                # remove other models that overlap this one
                for m in all_models.values():
                    m: PaperChainModel
                    if m.structure != model.structure:
                        continue
                    if m.atoms.intersects(model_atoms):
                        session.models.close([m])

            # add new models to open models list
            session.models.add([model], parent=model.structure)

        # make sure updated models are displayed
        model.display = True

        models.append(model)

    return models


@cmd(
    optional=[("atoms", AtomsArg)],
    keyword=[
        ("replace", BoolArg),
        ("update", BoolArg),
        ("start_end_centroid", BoolArg),
        ("rib_steps", IntArg),
        ("max_ring_size", IntArg),
        ("max_path_len", IntArg),
        ("rib_width", FloatArg),
        ("rib_height", FloatArg),
        ("colormap", EnumOf(("default", "norm"))),
        ("gum_twist", BoolArg),
    ],
    synopsis="Adds a Twister visualization to structures.",
)
def twister(
    session: Session,
    atoms: Atoms | None = None,
    replace=True,
    update=True,
    start_end_centroid=True,
    rib_steps=10,
    max_ring_size=10,
    max_path_len=5,
    rib_width=0.3,
    rib_height=0.05,
    colormap=None,
    gum_twist=False,
):
    """
    Adds a Twister visualization to structures.

    Args:
        structures: The structures to add the visualization to.
        replace: Whether to replace existing PaperChain models.
        update: Whether to automatically update the model if the
            structure changes.
        start_end_centroid: Whether to connect the Twister ribbons
            at the centroid of each ring.
        rib_steps: The number of steps used when rendering each ribbon.
        max_ring_size: A ring size limit when finding rings.
        max_path_len: A path length limit when finding linkages.
        rib_width: The ribbon width.
        rib_height: The ribbon height.
        colormap: Specify to use a colormap to set the ribbon color.
            One of 'default' or 'norm'.
        gum_twist: Whether to enable the Twister Gum variant.
    """

    atoms = check_atoms(atoms, session)

    if colormap is None:
        colormap = None
    elif colormap == "default":
        colormap = dihedral_colormap
    elif colormap == "norm":
        colormap = dihedral_norm_colormap
    else:
        raise ValueError(f"{colormap!r} is not a valid color map name")

    if replace:
        all_models = {m.atoms.hash(): m for m in session.models.list(type=TwisterModel)}
    else:
        all_models = {}

    models: list[TwisterModel] = []

    for structure, model_atoms in atoms.by_structure:
        model = all_models.get(model_atoms.hash())
        if model is None:
            name = f"{structure.name} Twister"
            model = TwisterModel(
                session,
                model_atoms,
                name,
                update=update,
                start_end_centroid=start_end_centroid,
                rib_steps=rib_steps,
                max_ring_size=max_ring_size,
                max_path_len=max_path_len,
                rib_width=rib_width,
                rib_height=rib_height,
                colormap=colormap,
                gum_twist=gum_twist,
            )
            new = True
        else:
            model.update_params(
                update=update,
                start_end_centroid=start_end_centroid,
                rib_steps=rib_steps,
                max_ring_size=max_ring_size,
                max_path_len=max_path_len,
                rib_width=rib_width,
                rib_height=rib_height,
                colormap=colormap,
                gum_twist=gum_twist,
            )
            new = False

        model.update()

        if new:
            if replace:
                # remove other models that overlap this one
                for m in all_models.values():
                    m: TwisterModel
                    if m.structure != model.structure:
                        continue
                    if m.atoms.intersects(model_atoms):
                        session.models.close([m])

            # add new models to open models list
            session.models.add([model], parent=model.structure)

        # make sure updated models are displayed
        model.display = True

        models.append(model)

    return models


@cmd(
    optional=[("atoms", AtomsArg)],
    keyword=[
        ("replace", BoolArg),
        ("update", BoolArg),
        ("max_ring_size", IntArg),
        ("max_path_len", IntArg),
        ("radius", FloatArg),
        ("colormap", EnumOf(("default", "norm"))),
        ("candy_cane", BoolArg),
        ("sphere_radius", FloatArg),
        ("sphere_colormap", EnumOf(("paperchain",))),
    ],
    synopsis="Adds a Strand visualization to structures.",
)
def strand(
    session: Session,
    atoms: Atoms | None = None,
    replace=True,
    update=True,
    max_ring_size=10,
    max_path_len=5,
    radius=0.75,
    colormap="default",
    candy_cane=False,
    sphere_radius=None,
    sphere_colormap=None,
):
    """
    Adds a Strand visualization to structures.

    Args:
        structures: The structures to add the visualization to.
        replace: Whether to replace existing PaperChain models.
        update: Whether to automatically update the model if the
            structure changes.
        max_ring_size: A ring size limit when finding rings.
        max_path_len: A path length limit when finding linkages.
        radius: The radius of the tubes.
        colormap: The colormap used to set the tube color.
        candy_cane: Whether to enable the Candy Cane variant.
            One of 'default' or 'norm'.
        sphere_radius: Specify to render spheres at each ring.
        sphere_colormap: The colormap used to set the sphere color.
            One of: 'paperchain'.
    """

    atoms = check_atoms(atoms, session)

    if colormap == "default":
        colormap = dihedral_colormap
    elif colormap == "norm":
        colormap = dihedral_norm_colormap
    else:
        raise ValueError(f"{colormap!r} is not a valid color map name")

    if sphere_radius is None:
        sphere_radius = radius

    if sphere_colormap == "paperchain":
        sphere_colormap = paperchain_colormap
    elif sphere_colormap is not None:
        raise ValueError(f"{sphere_colormap!r} is not a valid sphere color map name")

    if replace:
        all_models = {m.atoms.hash(): m for m in session.models.list(type=StrandModel)}
    else:
        all_models = {}

    models: list[StrandModel] = []

    for structure, model_atoms in atoms.by_structure:
        model = all_models.get(model_atoms.hash())
        if model is None:
            name = f"{structure.name} Strand"
            model = StrandModel(
                session,
                model_atoms,
                name,
                update=update,
                max_ring_size=max_ring_size,
                max_path_len=max_path_len,
                radius=radius,
                colormap=colormap,
                candy_cane=candy_cane,
                sphere_radius=sphere_radius,
                sphere_colormap=sphere_colormap,
            )
            new = True
        else:
            model.update_params(
                update=update,
                max_ring_size=max_ring_size,
                max_path_len=max_path_len,
                radius=radius,
                colormap=colormap,
                candy_cane=candy_cane,
                sphere_radius=sphere_radius,
                sphere_colormap=sphere_colormap,
            )
            new = False

        model.update()

        if new:
            if replace:
                # remove other models that overlap this one
                for m in all_models.values():
                    m: PaperChainModel
                    if m.structure != model.structure:
                        continue
                    if m.atoms.intersects(model_atoms):
                        session.models.close([m])

            # add new models to open models list
            session.models.add([model], parent=model.structure)

        # make sure updated models are displayed
        model.display = True

        models.append(model)

    return models


@cmd(
    required=[("bonds", Or(BondsArg, EmptyArg))],
    keyword=[
        ("colormap", EnumOf(("default", "norm"))),
        ("max_ring_size", IntArg),
        ("max_path_len", IntArg),
    ],
    synopsis="Color bonds by linkage dihedral angles.",
)
def color_bydihedral(
    session: Session,
    bonds: Bonds | None,
    colormap="default",
    max_ring_size=10,
    max_path_len=5,
):
    """
    Color bonds by linkage dihedral angles.

    Args:
        bonds: The bonds to color.
        colormap: The colormap to use. One of 'default' or 'norm'.
        max_ring_size: A ring size limit when finding rings.
        max_path_len: A path length limit when finding linkages.
    """

    if bonds is None:
        bonds = all_bonds(session)

    if colormap == "default":
        colormap = dihedral_colormap
    elif colormap == "norm":
        colormap = dihedral_norm_colormap
    else:
        raise ValueError(f"{colormap!r} is not a valid color map name")

    color_linkage_bonds(
        bonds,
        colormap,
        max_ring_size=max_ring_size,
        max_path_len=max_path_len,
    )


def check_atoms(atoms: Atoms | None, session: Session) -> Atoms:
    if atoms is None:
        atoms = all_atoms(session)
        if len(atoms) == 0:
            raise UserError("No atomic models open")
        setattr(atoms, "spec", "all atoms")
    elif len(atoms) == 0:
        msg = "No structures specified"
        if hasattr(atoms, "spec"):
            msg += f" by {getattr(atoms, 'spec')}"
        raise UserError(msg)
    return atoms
