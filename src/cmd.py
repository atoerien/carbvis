from weakref import WeakKeyDictionary

from chimerax.atomic import Structure, all_structures
from chimerax.core.commands import CmdDesc
from chimerax.core.session import Session

from .paperchain import PaperChainDrawing
from .twister import TwisterDrawing

paperchain_drawings: WeakKeyDictionary[Structure, PaperChainDrawing] = (
    WeakKeyDictionary()
)


def paperchain(session: Session):
    """An description"""

    for structure in all_structures(session):
        structure: Structure

        paperchain_drawing = None
        if structure in paperchain_drawings:
            paperchain_drawing = paperchain_drawings[structure]
            if paperchain_drawing.was_deleted:
                del paperchain_drawings[structure]
                paperchain_drawing = None

        if paperchain_drawing is not None:
            structure.remove_drawing(paperchain_drawing)
            del paperchain_drawings[structure]
        else:
            paperchain_drawing = PaperChainDrawing()
            paperchain_drawings[structure] = paperchain_drawing
            structure.add_drawing(paperchain_drawing)
            paperchain_drawing.compute_paperchain(structure)


paperchain_desc = CmdDesc(
    # required=[("atoms", Or(AtomsArg, EmptyArg))],
    # keyword=[("weighted", BoolArg), ("transformed", BoolArg)],
)


twister_drawings: WeakKeyDictionary[Structure, TwisterDrawing] = WeakKeyDictionary()


def twister(session: Session):
    """An description"""

    for structure in all_structures(session):
        structure: Structure

        twister_drawing = None
        if structure in twister_drawings:
            twister_drawing = twister_drawings[structure]
            if twister_drawing.was_deleted:
                del twister_drawings[structure]
                twister_drawing = None

        if twister_drawing is not None:
            structure.remove_drawing(twister_drawing)
            del twister_drawings[structure]
        else:
            twister_drawing = TwisterDrawing()
            twister_drawings[structure] = twister_drawing
            structure.add_drawing(twister_drawing)
            twister_drawing.compute_twister(structure)


twister_desc = CmdDesc(
    # required=[("atoms", Or(AtomsArg, EmptyArg))],
    # keyword=[("weighted", BoolArg), ("transformed", BoolArg)],
)
