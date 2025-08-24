import itertools

from chimerax.atomic import Bond, Bonds, Structure

from .carbs import dihedral_norm_colormap, find_linkages, find_rings
from .utils import color_float_to_ubyte


def color_bonds_bydihedral(bonds: Bonds, max_ring_size: int):
    for structure, bonds in bonds.by_structure:
        structure: Structure

        rings = find_rings(structure, max_ring_size)
        linkages = find_linkages(rings)

        for link in linkages:
            angles = link.calc_angles()
            color = color_float_to_ubyte(dihedral_norm_colormap(angles))

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
