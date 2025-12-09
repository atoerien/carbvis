import ast
import functools
from typing import Callable, ParamSpec, Protocol, TypeVar, cast

from chimerax.atomic import Atoms, Bonds, all_atoms, all_bonds
from chimerax.atomic.args import AtomsArg, BondsArg
from chimerax.core.colors import Color, Colormap
from chimerax.core.commands import (
    Annotation,
    AnnotationError,
    BoolArg,
    CmdDesc,
    ColorArg,
    ColormapArg,
    EmptyArg,
    EnumOf,
    FloatArg,
    IntArg,
    Or,
    commas,
)
from chimerax.core.errors import UserError
from chimerax.core.session import Session

from .carbs import (
    CarbLinkage,
    CarbRing,
    dihedral_colormap,
    dihedral_norm_colormap,
    dihedral_simple_colormap,
    paperchain_colormap,
)
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


def next_expr(text: str):
    try:
        node = ast.parse(text, mode="eval")
        return node, text, ""
    except SyntaxError as e:
        offset = e.offset
        if offset is None or offset == 0:
            raise
        offset -= 1
        if not text[offset - 1].isspace():
            raise

    while text[offset - 1].isspace():
        offset -= 1

    rest = text[offset:]
    text = text[:offset]
    node = ast.parse(text, mode="eval")
    return node, text, rest


def next_funclike_expr(text: str):
    try:
        node, text, rest = next_expr(text)
    except SyntaxError:
        raise AnnotationError("expected name, string or call") from None

    kwargs = {}

    body = node.body
    if isinstance(body, ast.Name):
        fn = body.id
    elif isinstance(body, ast.Constant) and type(body.value) is str:
        fn = body.value
    elif isinstance(body, ast.Call) and isinstance(body.func, ast.Name):
        fn = body.func.id
        if len(body.args) != 0:
            raise ValueError("may only use keyword arguments")
        for kw in body.keywords:
            arg = kw.arg
            try:
                value = ast.literal_eval(kw.value)
            except ValueError:
                raise ValueError("may only use literal argument values") from None
            kwargs[arg] = value
    else:
        raise AnnotationError("expected name, string or call")

    return (fn, kwargs), text, rest


class CarbColorArg(Annotation):
    name = "a color or a colormap"

    cmaps: dict[str, tuple[Callable, dict[str, tuple[type, ...]]]]

    @classmethod
    def parse(cls, text: str, session: Session):  # pyright: ignore[reportIncompatibleMethodOverride]
        try:
            return ColorArg.parse(text, session)
        except AnnotationError:
            pass

        try:
            (fn, kwargs), text, rest = next_funclike_expr(text)
        except AnnotationError:
            raise AnnotationError(f"expected {cls.name}") from None

        if fn not in cls.cmaps:
            raise ValueError(f"{fn!r} is not a valid color map")

        cmap, argdef = cls.cmaps[fn]

        partial_kwargs = {}
        for k, v in kwargs.items():
            if k not in argdef:
                expected = commas(repr(x) for x in argdef)
                raise ValueError(f"invalid arg {k!r}, expected {expected}")
            typ = argdef[k]

            if Colormap in typ:
                if v == "default":
                    # skip, don't forward to cmap
                    continue
                try:
                    v, _, _ = ColormapArg.parse(repr(v), session)
                except AnnotationError:
                    pass

            if not isinstance(v, typ):
                expected = commas(x.__name__ for x in typ)
                raise ValueError(f"expected {expected} for arg {k!r}")

            partial_kwargs[k] = v

        if partial_kwargs:
            f = functools.partial(cmap, **partial_kwargs)
        else:
            f = cmap
        return f, text, rest


RingColor = Callable[[CarbRing], Color] | Color


class RingColorArg(CarbColorArg):
    name = "a color or a ring colormap"

    cmaps: dict[str, tuple[Callable, dict[str, tuple[type, ...]]]] = {
        "paperchain": (
            paperchain_colormap,
            {},
        ),
    }


LinkageColor = Callable[[CarbLinkage], Color] | Color


class LinkageColorArg(CarbColorArg):
    name = "a color or a linkage colormap"

    cmaps: dict[str, tuple[Callable, dict[str, tuple[type, ...]]]] = {
        "default": (
            dihedral_colormap,
            {
                "palette": (Colormap,),
            },
        ),
        "simple": (
            dihedral_simple_colormap,
            {
                "palette": (Colormap,),
                "phi_min": (float, int),
                "phi_max": (float, int),
                "psi_min": (float, int),
                "psi_max": (float, int),
            },
        ),
        "norm": (
            dihedral_norm_colormap,
            {
                "palette": (Colormap,),
            },
        ),
    }


@cmd(
    optional=[("atoms", AtomsArg)],
    keyword=[
        ("replace", BoolArg),
        ("update", BoolArg),
        ("bipyramid_height", FloatArg),
        ("max_ring_size", IntArg),
        ("color", RingColorArg),
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
    color: RingColor = paperchain_colormap,
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
        color: How to color the polygons.
        tex_formula: The formula to use for the ring texture.
            One of 'stripes', 'grid', 'diamond', 'rings', 'waves'
        tex_period: The length of the texture period, in pixels.
        tex_duty: The fractional duty cycle of the repeating texture.
    """

    atoms = check_atoms(atoms, session)

    all_models: dict[bytes, PaperChainModel]
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
                cmap=color,
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
                cmap=color,
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
        ("color_top", LinkageColorArg),
        ("color_bottom", LinkageColorArg),
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
    color_top: LinkageColor = Color((0.9, 0.9, 0.9), mutable=False),
    color_bottom: LinkageColor = Color((0.5, 0.5, 1.0), mutable=False),
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
        color_top: How to color the top of the ribbons.
        color_bottom: How to color the bottom of the ribbons.
        gum_twist: Whether to enable the Twister Gum variant.
    """

    atoms = check_atoms(atoms, session)

    all_models: dict[bytes, TwisterModel]
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
                cmap_top=color_top,
                cmap_bottom=color_bottom,
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
                cmap_top=color_top,
                cmap_bottom=color_bottom,
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
        ("color", LinkageColorArg),
        ("candy_cane", BoolArg),
        ("sphere_radius", FloatArg),
        ("sphere_color", RingColorArg),
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
    color: LinkageColor = dihedral_colormap,
    candy_cane=False,
    sphere_radius=None,
    sphere_color: RingColor | None = None,
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
        color: How to color the tubes.
        candy_cane: Whether to enable the Candy Cane variant.
        sphere_radius: Specify to render spheres at each ring.
        sphere_color: How to color the spheres.
    """

    atoms = check_atoms(atoms, session)

    if sphere_radius is None:
        sphere_radius = radius

    all_models: dict[bytes, StrandModel]
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
                cmap=color,
                candy_cane=candy_cane,
                sphere_radius=sphere_radius,
                sphere_cmap=sphere_color,
            )
            new = True
        else:
            model.update_params(
                update=update,
                max_ring_size=max_ring_size,
                max_path_len=max_path_len,
                radius=radius,
                cmap=color,
                candy_cane=candy_cane,
                sphere_radius=sphere_radius,
                sphere_cmap=sphere_color,
            )
            new = False

        model.update()

        if new:
            if replace:
                # remove other models that overlap this one
                for m in all_models.values():
                    m: StrandModel
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
        ("color", LinkageColorArg),
        ("max_ring_size", IntArg),
        ("max_path_len", IntArg),
    ],
    synopsis="Color bonds by linkage dihedral angles.",
)
def color_bydihedral(
    session: Session,
    bonds: Bonds | None,
    color: LinkageColor = dihedral_colormap,
    max_ring_size=10,
    max_path_len=5,
):
    """
    Color bonds by linkage dihedral angles.

    Args:
        bonds: The bonds to color.
        color: How to color the bonds.
        max_ring_size: A ring size limit when finding rings.
        max_path_len: A path length limit when finding linkages.
    """

    if bonds is None:
        bonds = all_bonds(session)

    color_linkage_bonds(
        bonds,
        color,
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
