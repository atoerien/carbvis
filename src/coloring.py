import itertools
from typing import Callable

from chimerax.atomic import Bond, Bonds, Structure

from .carbs import CarbLinkage, find_linkages, find_rings
from .utils import FloatArray, color_float_to_ubyte


def color_linkage_bonds(
    bonds: Bonds,
    colormap: Callable[[CarbLinkage], FloatArray],
    *,
    max_ring_size: int,
    max_path_len: int,
):
    """
    Set the color of linkage bonds using a colormap.

    Args:
        bonds: The bonds to color.
        max_ring_size: A ring size limit when finding rings.
        max_path_len: A path length limit when finding linkages.
        colormap: The colormap to use, mapping a CarbLinkage to an
            RGB [0, 1] float array.
    """

    for structure, bonds in bonds.by_structure:
        structure: Structure

        rings = find_rings(structure.atoms, max_ring_size)
        linkages = find_linkages(rings, max_path_len)

        for link in linkages:
            color = color_float_to_ubyte(colormap(link))

            for atom in link.atoms[1:-1]:
                atom.color = color

            for a1, a2 in itertools.pairwise(link.atoms):
                bond: Bond
                for bond in a1.bonds:
                    if bond.other_atom(a1) == a2:
                        break
                else:
                    continue

                bond.color = color
                bond.halfbond = False
