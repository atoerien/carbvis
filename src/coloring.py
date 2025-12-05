import itertools
from typing import Callable

from chimerax.atomic import Bond, Bonds, Structure
from chimerax.core.colors import Color

from .carbs import CarbLinkage, find_linkages, find_rings


def color_linkage_bonds(
    bonds: Bonds,
    cmap: Callable[[CarbLinkage], Color] | Color,
    *,
    max_ring_size: int,
    max_path_len: int,
):
    """
    Set the color of linkage bonds.

    Args:
        bonds: The bonds to color.
        max_ring_size: A ring size limit when finding rings.
        max_path_len: A path length limit when finding linkages.
        color: The color to use.
    """

    for structure, bonds in bonds.by_structure:
        structure: Structure

        rings = find_rings(structure.atoms, max_ring_size)
        linkages = find_linkages(rings, max_path_len)

        for link in linkages:
            if callable(cmap):
                color = cmap(link)
            else:
                color = cmap
            color = color.uint8x4()

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
