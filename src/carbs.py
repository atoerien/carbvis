from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from chimerax.atomic import Atom, Atoms, Element


@dataclass
class SmallRing:
    """
    A SmallRing contains a list of atoms which are connected
    to each other to form a loop.  The atom numbers are the
    unique atom numbers as used in BaseMolecule. The ordering of
    the atoms, in addition to specifying how the atoms in the ring are
    connected, also gives the orientation (handedness) of the ring
    if orientated is non-zero.
    """

    atoms: list[Atom] = field(default_factory=lambda: [])
    orientated: bool = field(default=False)

    def copy(self):
        return SmallRing(
            atoms=self.atoms.copy(),
            orientated=self.orientated,
        )

    def get_ring_coords(self):
        n = len(self.atoms)

        ret = np.empty((n, 3), dtype=np.float32)

        for i, atom in enumerate(self.atoms):
            ret[i] = atom.coord

        return ret


@dataclass
class LinkagePath:
    """
    A Linkage Path object consists of
     - a list of atoms in the path
     - the index of the start and end rings
    """

    atoms: list[Atom] = field(default_factory=lambda: [])
    start_ring: int = field(default=-1)
    end_ring: int = field(default=-1)

    def copy(self):
        return LinkagePath(
            atoms=self.atoms.copy(),
            start_ring=self.start_ring,
            end_ring=self.end_ring,
        )


@dataclass(unsafe_hash=True)
class LinkageEdge:
    left_atom: Final[Atom]
    right_atom: Final[Atom]

    def __init__(self, atom_left: Atom, atom_right: Atom):
        if atom_left > atom_right:
            self.left_atom = atom_right
            self.right_atom = atom_left
        else:
            self.left_atom = atom_left
            self.right_atom = atom_right


@dataclass
class SmallRingLinkages:
    links: list[LinkageEdge] = field(default_factory=lambda: [])
    paths: list[LinkagePath] = field(default_factory=lambda: [])

    _edges_to_paths: dict[LinkageEdge, list[LinkagePath]] = field(
        default_factory=lambda: {}
    )

    def clear(self):
        self.links.clear()
        self.paths.clear()
        self._edges_to_paths.clear()

    def add_linkage_path(self, lp: LinkagePath):
        atom_right = lp.atoms[0]

        for i in range(1, len(lp.atoms)):
            atom_left = atom_right
            atom_right = lp.atoms[i]
            edge = LinkageEdge(atom_left, atom_right)

            if edge in self._edges_to_paths:
                self._edges_to_paths[edge].append(lp)
            else:
                self._edges_to_paths[edge] = [lp]

        self.paths.append(lp)

    def shares_linkage_edges(self, lp: LinkagePath):
        atom_right = lp.atoms[0]

        for i in range(1, len(lp.atoms)):
            atom_left = atom_right
            atom_right = lp.atoms[i]
            edge = LinkageEdge(atom_left, atom_right)

            if edge in self._edges_to_paths and len(self._edges_to_paths[edge]) > 1:
                return True

        return False


def find_small_rings(atoms: Atoms, maxringsize: int):
    # (src, dest)
    back_edges = find_back_edges(atoms)
    rings = find_small_rings_from_back_edges(atoms, back_edges, maxringsize)

    print(f"BACK EDGES: {len(back_edges)}")
    # for src, dest in back_edges:
    #     print(f"  SRC: {src}, DST: {dest}")

    print(f"SMALL RINGS: {len(rings)}")
    # for ring in rings:
    #     print(f"  RING: {ring}")

    orientate_small_rings(rings)

    n_orientated_rings = 0
    for ring in rings:
        if ring.orientated:
            n_orientated_rings += 1
    print(f"RINGS ORIENTATED: {n_orientated_rings}")

    return rings


INTREE_NOT = -1  # not in tree
INTREE_NOPARENT = -2  # no parent


def find_back_edges(atoms: Atoms):
    back_edges: list[tuple[Atom, Atom]] = []

    intree_parents: dict[Atom, Atom | None] = {}

    for atom in atoms:
        if atom not in intree_parents:  # not been visited
            find_connected_subgraph_back_edges(
                atom,
                back_edges,
                intree_parents,
            )

    return back_edges


def find_connected_subgraph_back_edges(
    atom: Atom,
    back_edges: list[tuple[Atom, Atom]],
    intree_parents: dict[Atom, Atom | None],
):
    node_stack = [atom]
    intree_parents[atom] = None  # in tree, but doesn't have parent

    while node_stack:
        cur_atom = node_stack.pop()
        parent_atom = intree_parents[cur_atom]

        for child_atom in cur_atom.neighbors:
            child_atom: Atom
            if child_atom in intree_parents:
                # back-edge found
                if child_atom != parent_atom and id(child_atom) > id(cur_atom):
                    # we ignore edges back to the parent
                    # and only add each back edge once
                    # (it'll crop up twice since each bond is listed on both atoms)
                    back_edges.append((cur_atom, child_atom))
            else:
                # extended tree
                intree_parents[child_atom] = cur_atom
                node_stack.append(child_atom)


def find_small_rings_from_back_edges(
    atoms: Atoms,
    back_edges: list[tuple[Atom, Atom]],
    maxringsize: int,
):
    rings: list[SmallRing] = []

    used_edges: set[tuple[Atom, Atom]] = set()
    used_atoms: set[Atom] = set()

    # cap the peak number of rings to find based on the size of the
    # input structure. This should help prevent unusual structures
    # with very high connectivity, such as silicon nanodevices from
    # blowing up the ring search code
    max_rings = 2000 + int(100.0 * (len(atoms) ** 0.5))

    for back_edge in back_edges:
        src, dest = back_edge
        ring = SmallRing(atoms=[src, dest])

        # first atom is not marked used, since we're allowed to re-use it.
        used_atoms.add(dest)

        find_small_rings_from_partial(
            rings,
            ring,
            maxringsize,
            used_edges,
            used_atoms,
        )

        if len(rings) > max_rings:
            # TODO: log
            print(
                f"Maximum number of rings ({max_rings}) exceeded. "
                f"Stopped looking for rings after {len(rings)} rings found."
            )
            break

        used_edges.add(back_edge)

        # remove last atom from used_atoms
        used_atoms.remove(dest)

    return rings


def find_small_rings_from_partial(
    rings: list[SmallRing],
    ring: SmallRing,
    maxringsize: int,
    used_edges: set[tuple[Atom, Atom]],
    used_atoms: set[Atom],
):
    atom_stack: list[Atom] = [ring.atoms[-1]]
    bond_pos_stack: list[int] = [0]

    while atom_stack:
        cur_atom = atom_stack.pop()
        cur_atom_neighbors: list[Atom] = cur_atom.neighbors
        next_bond_pos = bond_pos_stack.pop()

        do_pop = False  # whether we need to pop the current atom

        if len(ring.atoms) > maxringsize:
            do_pop = True

        if next_bond_pos == 0 and not do_pop:
            barred = False
            closes = False

            for child_atom in cur_atom_neighbors:
                # check that this is not an edge immediately back to the previous atom
                if child_atom == ring.atoms[-2]:
                    continue

                # check that this is not an atom we've included
                # (an exception is the first atom, which we're allowed to try add, obviously :)
                if child_atom in used_atoms:
                    barred = True
                    continue

                # check is this not a back edge which has already been used
                if (cur_atom, child_atom) in used_edges:
                    if child_atom == ring.atoms[0]:
                        barred = True  # used back-edge which closes the ring counts as barred
                    continue

                # check whether ring closes
                if child_atom == ring.atoms[0]:
                    closes = True
                    continue

            if closes and not barred:
                rings.append(ring.copy())

            if closes or barred:
                # adding this atom would create a barred ring, so skip to next atom on stack
                do_pop = True

        if not do_pop:
            for i in range(next_bond_pos, len(cur_atom_neighbors)):
                child_atom = cur_atom_neighbors[i]

                # check that this is not an edge immediately back to the previous atom
                if child_atom == ring.atoms[-2]:
                    continue

                # check is this not a back edge which has already been used
                if (cur_atom, child_atom) in used_edges:
                    continue

                # Append child atom and go deeper
                ring.atoms.append(child_atom)
                used_atoms.add(child_atom)

                # push current state and new state onto stack and recurse
                atom_stack.append(cur_atom)
                bond_pos_stack.append(i + 1)
                atom_stack.append(child_atom)
                bond_pos_stack.append(0)
                break
            else:
                # we finished the children, pop one parent
                do_pop = True

        if do_pop and atom_stack:
            # finished with cur_atom and all its children
            # clean up before returning from recurse
            ring.atoms.pop()
            used_atoms.remove(cur_atom)


def orientate_small_rings(rings: list[SmallRing]):
    for ring in rings:
        oxygen = -1

        # Find an oxygen (or something with two bonds)
        for i, atom in enumerate(ring.atoms):
            element: Element = atom.element
            if element.number == 8 or atom.num_bonds == 2:
                oxygen = i
                break

        if oxygen == -1:
            continue

        # find atoms before and after oxygen (taking into account wrapping)

        atom_before_O = ring.atoms[oxygen - 1]

        if oxygen == len(ring.atoms) - 1:
            atom_after_O = ring.atoms[0]
        else:
            atom_after_O = ring.atoms[oxygen + 1]

        # ensure C1 carbon is after the oxygen
        # leave unorientated if the C1 carbon can't be found
        if (
            atom_before_O.name == "C1"
            or atom_before_O.name == "C1'"
            or atom_before_O == "C_1"
            or atom_before_O.name == "C2"
            or atom_before_O.name == "C2'"
            or atom_before_O == "C_2"
        ):
            ring.atoms.reverse()
            ring.orientated = True
        elif (
            atom_after_O.name == "C1"
            or atom_after_O.name == "C1'"
            or atom_after_O == "C_1"
            or atom_after_O.name == "C2"
            or atom_after_O.name == "C2'"
            or atom_after_O == "C_2"
        ):
            ring.orientated = True


def find_small_ring_linkages(rings: list[SmallRing]):
    linkages = SmallRingLinkages()

    atom_to_ring: dict[Atom, int] = {}
    multi_ring_atoms: set[Atom] = set()
    used_atoms: set[Atom] = set()

    for i, ring in enumerate(rings):
        # XXX: Uncomment this if we want to include non-orientated rings
        #      in linkage paths
        # if not ring.orientated:
        #     continue
        for atom in ring.atoms:
            if atom in atom_to_ring:
                multi_ring_atoms.add(atom)
            else:
                atom_to_ring[atom] = i

    for i, ring in enumerate(rings):
        if not ring.orientated:
            continue
        for atom in ring.atoms:
            if atom in multi_ring_atoms:
                continue
            lp = LinkagePath(atoms=[atom], start_ring=i)
            find_linkages_for_ring_from_partial(
                rings,
                linkages,
                lp,
                atom_to_ring,
                multi_ring_atoms,
                used_atoms,
            )

    return linkages


def find_linkages_for_ring_from_partial(
    rings: list[SmallRing],
    linkages: SmallRingLinkages,
    lp: LinkagePath,
    atom_to_ring: dict[Atom, int],
    multi_ring_atoms: set[Atom],
    used_atoms: set[Atom],
):
    atom_stack: list[Atom] = [lp.atoms[-1]]
    bond_pos_stack: list[int] = [0]

    while atom_stack:
        cur_atom = atom_stack.pop()
        cur_atom_neighbors: list[Atom] = cur_atom.neighbors
        next_bond_pos = bond_pos_stack.pop()

        if next_bond_pos == 0:
            used_atoms.add(cur_atom)

        for i in range(next_bond_pos, len(cur_atom_neighbors)):
            child_atom = cur_atom_neighbors[i]

            # check that this isn't an atom that belongs to multiple rings
            if child_atom in multi_ring_atoms:
                continue

            # check that this is not an edge immediately back to the previous atom
            # (when there is only one atom in the path, it can't be a link back)
            if len(lp.atoms) > 1 and child_atom == lp.atoms[-2]:
                continue

            if child_atom in atom_to_ring:
                ring_idx = atom_to_ring[child_atom]
                ring = rings[ring_idx]

                # check that we haven't arrived at a non-orientated ring
                if not ring.orientated:
                    continue

                # only store paths from smaller ringidx to larger ringidx (to avoid getting a copy of each orientation of the path)
                # ignore paths which return to the same ring
                # check that we're leaving the starting ring
                if ring_idx <= lp.start_ring:
                    continue

                lp.atoms.append(child_atom)
                lp.end_ring = ring_idx
                linkages.add_linkage_path(lp.copy())
                lp.end_ring = -1
                lp.atoms.pop()
                continue
            else:
                # check that this is not an atom we've included
                # (an exception is the first atom, which we're allowed to try add, obviously :)
                if child_atom in used_atoms:
                    continue

                lp.atoms.append(child_atom)

                # push current state and new state onto stack and recurse
                atom_stack.append(cur_atom)
                bond_pos_stack.append(i + 1)
                atom_stack.append(child_atom)
                bond_pos_stack.append(0)
                break
        else:
            if atom_stack:
                # clean up before returning from recurse
                lp.atoms.pop()
                used_atoms.remove(cur_atom)


def get_ring_centroid_and_normal(ring_coords: np.ndarray):
    n = ring_coords.shape[0]

    centroid = np.zeros(3, dtype=np.float32)
    normal = np.zeros(3, dtype=np.float32)

    # calculate centroid and normal
    for i in range(n):
        # calculate next ring position (wrapping as necessary)
        next_i = i + 1
        if next_i >= n:
            next_i = 0

        curvec = ring_coords[i]
        nextvec = ring_coords[next_i]

        # update centroid
        centroid += curvec

        # update normal (this is Newell's method; see Carbohydra paper)
        normal[0] += (curvec[1] - nextvec[1]) * (curvec[2] + nextvec[2])
        normal[1] += (curvec[2] - nextvec[2]) * (curvec[0] + nextvec[0])
        normal[2] += (curvec[0] - nextvec[0]) * (curvec[1] + nextvec[1])

    centroid *= 1.0 / n
    normal /= np.linalg.norm(normal)

    return centroid, normal
