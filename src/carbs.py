from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from chimerax.atomic import Atom, Atoms, Element, Structure
from chimerax.geometry import dihedral

from .utils import FloatArray, Frame, dfs_paths


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

    def calc_pucker_amplitude(self):
        """
        Calculate the Cremer-Pople puckering amplitude for this ring.
        """

        # get centred ring coords
        coords = self.coords - self.get_centroid()
        n = coords.shape[0]

        # Calculate cartesian axes based on coords of nuclei in ring
        # using cremer-pople algorithm. It is assumed that the
        # centre of geometry is the centre of the ring.

        indices = np.arange(n)
        ze_angle = 2.0 * np.pi * (indices - 1) / n
        ze_sin = np.sin(ze_angle)
        ze_cos = np.cos(ze_angle)

        Rp = np.sum(coords.T * ze_sin, axis=1)
        Rpp = np.sum(coords.T * ze_cos, axis=1)

        z = np.cross(Rp, Rpp)
        z /= np.linalg.norm(z)

        # lmbda = np.dot(z, Rp)
        # y = Rp - z * lmbda
        # y /= np.linalg.norm(y)

        # x = np.cross(y, z)

        # calculate displacement from mean plane
        displ = np.dot(coords, z)

        q = np.sqrt(np.sum(displ**2))
        return min(q, 2.0)  # truncate amplitude at 2


@dataclass
class CarbLinkage:
    """
    A (C1->Cx) linkage between two rings.
    start_ring -> *atoms -> end_ring
    """

    atoms: list[Atom]
    start_ring: CarbRing
    end_ring: CarbRing

    def calc_angles(self) -> FloatArray:
        n = len(self.atoms)

        if n < 2:
            return np.zeros(0, dtype=np.float32)

        atoms = self.atoms
        start_ring = self.start_ring
        end_ring = self.end_ring

        angle_atoms = []

        # add side-atom before first atom
        found = False
        first_atom = atoms[0]
        for a in first_atom.neighbors:
            if a != atoms[1] and a not in start_ring.atoms:
                if found:
                    print(f"warning: multiple non-ring atoms attached to {first_atom}")
                else:
                    angle_atoms.append(a)
                    found = True
        if not found:
            # print(f"no non-ring atom attached to {first_atom}")
            return np.zeros(0, dtype=np.float32)

        angle_atoms.extend(atoms)

        # add side-atom after last atom
        found = False
        last_atom = atoms[-1]
        for a in last_atom.neighbors:
            if a != atoms[-2] and a not in end_ring.atoms:
                if found:
                    print(f"warning: multiple non-ring atoms attached to {last_atom}")
                else:
                    angle_atoms.append(a)
                    found = True
        if not found:
            # print(f"no non-ring atom attached to {first_atom}")
            return np.zeros(0, dtype=np.float32)

        angles = np.empty(n - 1, dtype=np.float32)
        for i in range(n - 1):
            angles[i] = dihedral(*(a.coord for a in angle_atoms[i : i + 4]))
        return angles


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
    max_size: int,
    orientate: bool = True,
) -> list[CarbRing]:
    rings = structure.rings(all_size_threshold=max_size)
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

    def get_neighbors(atom: Atom):
        if atom is start_atom:
            # for the first atom, only allow edges out of the ring
            ret = [a for a in atom.neighbors if a not in atom_to_ring]
        elif atom in atom_to_ring:
            # otherwise if we're in a ring it's an endpoint
            ret = []
        else:
            ret = [a for a in atom.neighbors]
        return ret

    linkages: list[CarbLinkage] = []

    # keep the same visited set through all the dfs_paths calls
    # start_atom then stays in the set, preventing duplicate
    # forward and reverse paths
    visited: set[int] = set()

    for start_ring in rings:
        if not start_ring.orientated:
            continue

        for start_atom in start_ring:
            if start_atom in multi_ring_atoms:
                continue

            id_start_atom = id(start_atom)
            if id_start_atom in visited:
                continue
            visited.add(id_start_atom)

            for linkage in dfs_paths(get_neighbors, start_atom, visited):
                end_atom = linkage[-1]
                if start_atom is end_atom:
                    continue

                if end_atom in multi_ring_atoms:
                    continue

                if end_atom not in atom_to_ring:
                    continue
                end_ring = atom_to_ring[end_atom]

                # enforce ordering, start_ring should be the C1 carbon
                # also if no C1 involved at all, start_ring can be the C2 carbon
                if start_atom.name in ("C1", "C1'", "C_1"):
                    linkages.append(CarbLinkage(linkage, start_ring, end_ring))
                elif end_atom.name in ("C1", "C1'", "C_1"):
                    linkage.reverse()
                    linkages.append(CarbLinkage(linkage, end_ring, start_ring))
                elif start_atom.name in ("C2", "C2'", "C_2"):
                    linkages.append(CarbLinkage(linkage, start_ring, end_ring))
                elif end_atom.name in ("C2", "C2'", "C_2"):
                    linkage.reverse()
                    linkages.append(CarbLinkage(linkage, end_ring, start_ring))
                else:
                    print(
                        f"warning: linkage {start_atom}->{end_atom} is not a (C1->Cx) linkage"
                    )
                    linkages.append(CarbLinkage(linkage, start_ring, end_ring))

    # print(f"LINKAGES: {len(linkages)}")
    return linkages


def find_chains(linkages: list[CarbLinkage]) -> list[CarbChain]:
    # directed graph
    graph: dict[int, CarbLinkage] = {}
    ring_has_edges_in: set[int] = set()
    for linkage in linkages:
        id_start_ring = id(linkage.start_ring)
        if id_start_ring in graph:
            print(f"warning: ignoring multiple edges out of ring {linkage.start_ring}")
        else:
            graph[id_start_ring] = linkage

        ring_has_edges_in.add(id(linkage.end_ring))

    entrypoints: list[CarbRing] = []
    for linkage in linkages:
        start_ring = linkage.start_ring
        if id(start_ring) not in ring_has_edges_in:
            entrypoints.append(start_ring)

    # graph should be a tree, so no entrypoints means no linkages
    if not entrypoints:
        return []

    def get_neighbors(ring: CarbRing):
        id_ring = id(ring)
        if id_ring in graph:
            return (graph[id_ring].end_ring,)
        else:
            return ()

    chains: list[CarbChain] = []

    for entrypoint in entrypoints:
        for chain in dfs_paths(get_neighbors, entrypoint):
            paths = [graph[id(r)] for r in chain[:-1]]
            chains.append(CarbChain(chain, paths))

    chains.sort(key=lambda c: len(c.rings), reverse=True)

    visited: set[int] = set()
    for chain in chains:
        for i, ring in enumerate(chain.rings):
            id_ring = id(ring)
            if id_ring in visited:
                del chain.rings[i + 1 :]
                del chain.linkages[i:]
                break
            visited.add(id_ring)

    # print(f"CHAINS: {len(chains)}")
    return chains


def paperchain_colormap(pucker: float) -> FloatArray:
    rgb = np.zeros(3, dtype=np.float32)  # default color is black

    # Hot to cold color map:
    # Red -> Yellow -> Green -> Cyan -> Blue -> Magenta
    if pucker < 0.40:
        # Red (1,0,0) -> Yellow (1,1,0)
        rgb[0] = 1.0  # red
        rgb[1] = pucker * 2.5  # increase green -> yellow
        rgb[2] = 0.0
    elif pucker < 0.56:
        # Yellow (1,1,0) -> Green (0,1,0)
        rgb[0] = 1.0 - (pucker - 0.40) * 6.25  # decrease red -> green
        rgb[1] = 1.0
        rgb[2] = 0.0
    elif pucker < 0.64:
        # Green (0,1,0) -> Cyan (0,1,1)
        rgb[0] = 0.0
        rgb[1] = 1.0  # green
        rgb[2] = (pucker - 0.56) * 12.5  # increase blue
    elif pucker < 0.76:
        # Cyan (0,1,1) -> Blue (0,0,1)
        rgb[0] = 0.0
        rgb[1] = 1.0 - (pucker - 0.64) * 5.0  # decrease green
        rgb[2] = 1.0
    else:
        # Blue (0,0,1) -> Magenta (1,0,1)
        rgb[0] = (pucker - 0.76) * 0.8  # increase red
        rgb[1] = 0.0
        rgb[2] = 1.0

    return rgb


def dihedral_norm_colormap(angles: FloatArray) -> FloatArray:
    if angles.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)

    v = np.linalg.norm(angles)
    rgb = np.empty(3, dtype=np.float32)

    v /= 180
    rgb[0] = 1
    rgb[1] = 1 - v
    rgb[2] = 1 - v

    return rgb
