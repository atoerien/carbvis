from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from chimerax.atomic import Atom, Atoms, Element, Structure

from .utils import FloatArray, Frame


@dataclass
class CarbRing:
    """
    A ring of atoms connected to each other in a loop.
    If orientated is True, the order of atoms gives the
    orientation (handedness) of the ring.
    """

    atoms: Atoms
    orientated: bool = field(default=False)

    def __getitem__(self, i: int) -> Atom:
        return self.atoms[i]  # pyright: ignore[reportReturnType]

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)

    @property
    def coords(self) -> FloatArray:
        return self.atoms.coords

    def orientate(self) -> bool:
        oxygen = -1

        # Find an oxygen (or something with two bonds)
        for i, atom in enumerate(self):
            element: Element = atom.element
            if element.number == 8 or atom.num_bonds == 2:
                oxygen = i
                break

        if oxygen == -1:
            return False

        # find atoms before and after oxygen (taking into account wrapping)

        atom_before_O = self[oxygen - 1]

        if oxygen == len(self.atoms) - 1:
            atom_after_O = self[0]
        else:
            atom_after_O = self[oxygen + 1]

        # ensure C1 carbon is after the oxygen
        # leave unorientated if the C1 carbon can't be found
        if atom_before_O.name in ("C1", "C1'", "C_1", "C2", "C2'", "C_2"):
            # reverse atom list
            self.atoms = Atoms(np.flip(self.atoms.pointers))
            self.orientated = True
            return True
        elif atom_after_O.name in ("C1", "C1'", "C_1", "C2", "C2'", "C_2"):
            self.orientated = True
            return True
        else:
            return False

    def get_centroid(self) -> FloatArray:
        return np.mean(self.coords, axis=0)

    def get_centroid_and_normal(self) -> tuple[FloatArray, FloatArray]:
        ring_coords = self.coords

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

        centroid /= n

        # flip while normalizing - the "up" of a carbohydrate ring is a LH normal
        # but Newell's method gives a RH normal
        normal /= -np.linalg.norm(normal)

        return centroid, normal

    def get_frame(self) -> Frame:
        centroid, up = self.get_centroid_and_normal()

        # use the first atom as forward, should not be parallel to up
        forward = self[0].coord - centroid
        forward -= np.dot(forward, up) * up
        if np.allclose(forward, 0.0):
            # fallback: pick a world axis
            if abs(up[0]) < 0.9:
                forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            forward = forward - np.dot(forward, up) * up
        forward /= np.linalg.norm(forward)

        # up and forward are length 1, no need to norm
        right = np.cross(up, forward)

        return Frame(centroid, forward, right, up)


@dataclass
class CarbLinkage:
    """
    A (C1->Cx) linkage between two rings.
    start_ring -> *atoms -> end_ring
    """

    atoms: list[Atom]
    start_ring: CarbRing
    end_ring: CarbRing


@dataclass
class CarbChain:
    """
    A chain of rings, connected by linkages.
    linkages[i] = rings[i] -> rings[i+1]
    """

    rings: list[CarbRing]
    linkages: list[CarbLinkage]


def find_rings(
    structure: Structure,
    maxringsize: int,
    orientate: bool = True,
) -> list[CarbRing]:
    rings = structure.rings(all_size_threshold=maxringsize)
    rings = [CarbRing(ring.ordered_atoms) for ring in rings]

    # print(f"RINGS: {len(rings)}")
    # for ring in rings:
    #     print(f"  RING: {ring}")

    if orientate:
        n_orientated_rings = 0
        for ring in rings:
            if ring.orientate():
                n_orientated_rings += 1
        # print(f"ORIENTATED RINGS: {n_orientated_rings}")

    return rings


def find_linkages(rings: list[CarbRing]) -> list[CarbLinkage]:
    atom_to_ring: dict[Atom, CarbRing] = {}
    multi_ring_atoms: set[Atom] = set()

    for start_ring in rings:
        for atom in start_ring:
            if atom in atom_to_ring:
                multi_ring_atoms.add(atom)
            else:
                atom_to_ring[atom] = start_ring

    linkages: list[CarbLinkage] = []

    used_atoms: set[Atom] = set()

    for ring in rings:
        if not ring.orientated:
            continue
        for atom in ring:
            if atom in multi_ring_atoms:
                continue
            find_linkages_from_atom(
                linkages,
                ring,
                atom,
                atom_to_ring,
                multi_ring_atoms,
                used_atoms,
            )

    # print(f"LINKAGES: {len(linkages.paths)}")
    return linkages


def find_linkages_from_atom(
    linkages: list[CarbLinkage],
    start_ring: CarbRing,
    start_atom: Atom,
    atom_to_ring: dict[Atom, CarbRing],
    multi_ring_atoms: set[Atom],
    used_atoms: set[Atom],
):
    atom_stack: list[Atom] = [start_atom]
    bond_pos_stack: list[int] = [0]

    while atom_stack:
        cur_atom = atom_stack.pop()
        cur_atom_neighbors: list[Atom] = cur_atom.neighbors
        next_bond_pos = bond_pos_stack.pop()

        if next_bond_pos == 0:
            used_atoms.add(cur_atom)

        for i in range(next_bond_pos, len(cur_atom_neighbors)):
            next_atom = cur_atom_neighbors[i]

            # check that this isn't an atom that belongs to multiple rings
            if next_atom in multi_ring_atoms:
                continue

            # check that this is not an edge immediately back to the previous atom
            # (when there is only one atom in the path, it can't be a link back)
            if atom_stack and next_atom == atom_stack[-1]:
                continue

            if next_atom in atom_to_ring:
                end_ring = atom_to_ring[next_atom]

                # check that we haven't arrived at a non-orientated ring
                if not end_ring.orientated:
                    continue

                # only store paths from smaller ringidx to larger ringidx (to avoid getting a copy of each orientation of the path)
                # ignore paths which return to the same ring
                # check that we're leaving the starting ring
                if id(end_ring) <= id(start_ring):
                    continue

                atoms = atom_stack.copy()
                atoms.append(cur_atom)
                atoms.append(next_atom)
                linkages.append(CarbLinkage(atoms, start_ring, end_ring))
                continue
            else:
                # check that this is not an atom we've included
                # (an exception is the first atom, which we're allowed to try add, obviously :)
                if next_atom in used_atoms:
                    continue

                # push current state and new state onto stack and recurse
                atom_stack.append(cur_atom)
                bond_pos_stack.append(i + 1)
                atom_stack.append(next_atom)
                bond_pos_stack.append(0)
                break
        else:
            if atom_stack:
                # clean up before returning from recurse
                used_atoms.remove(cur_atom)
