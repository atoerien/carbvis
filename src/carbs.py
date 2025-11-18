from dataclasses import dataclass, field
from typing import Iterator, Self

import numpy as np
from chimerax.atomic import Atom, Atoms, Element, Residue, Residues, Ring, Structure
from chimerax.geometry import dihedral

from .utils import FloatArray, Frame, dfs_paths, gaussian


@dataclass
class CarbRing:
    """
    A ring of atoms connected to each other in a loop.

    Indexing, length and iteration operations are forwarded to the
    atoms list.

    Attributes:
        atoms: The atoms in the ring.
        residue: The residue the atoms are part of.
        orientation: Indicates if the order of the atoms corresponds
            to the orientation (handedness) of the ring.
    """

    atoms: Atoms
    residue: Residue
    orientated: bool = field(default=False)

    @classmethod
    def from_ring(cls, ring: Ring) -> Self:
        """
        Create an instance from a Ring object.

        The ring must not cross a residue boundary.
        """

        atoms: Atoms = ring.ordered_atoms

        # roll the atoms list for a consistent ring ordering
        roll_shift = -np.argmin([a.name for a in atoms])
        if roll_shift != 0:
            atoms = Atoms(np.roll(atoms.pointers, roll_shift))

        residues: Residues = atoms.unique_residues
        if len(residues) != 1:
            raise ValueError("Ring crosses residue boundary")

        ret = cls(atoms, residues[0])

        ret.orientate()
        # if not ret.orientate():
        #     print(f"warning: could not orientate ring {ret}")

        return ret

    def __getitem__(self, i: int) -> Atom:
        return self.atoms[i]  # pyright: ignore[reportReturnType]

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)

    @property
    def coords(self) -> FloatArray:
        """The atom coordinates."""
        return self.atoms.coords

    def orientate(self) -> bool:
        """
        Orientate the ring, flipping the atom list such that the
        order corresponds to the handedness of the ring.

        Returns:
            Whether the ring was able to be oriented or not.
        """

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
        atom_after_O = self[(oxygen + 1) % len(self.atoms)]

        # ensure C1 carbon is after the oxygen
        # leave unorientated if the C1 carbon can't be found
        if atom_before_O.name in ("C1", "C1'", "C_1", "C2", "C2'", "C_2"):
            # reverse atom list, while keeping the first atom in the same place
            ptrs = self.atoms.pointers
            ptrs[1:] = np.flip(ptrs[1:])
            self.atoms = Atoms(ptrs)
            self.orientated = True
            return True
        elif atom_after_O.name in ("C1", "C1'", "C_1", "C2", "C2'", "C_2"):
            self.orientated = True
            return True
        else:
            return False

    def get_centroid(self) -> FloatArray:
        """Calculate the ring centroid."""
        return np.mean(self.coords, axis=0)

    def get_centroid_and_normal(self) -> tuple[FloatArray, FloatArray]:
        """
        Calculate the ring centroid and normal vector.

        The returned normal vector points in the direction of the
        "top" of the ring (a left-handed normal).

        Returns:
            A tuple (centroid, normal).
        """

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
        """
        Return a Frame corresponding to the plane of the ring.

        The forward vector of the frame points towards the first atom,
        and the up vector is the ring normal.
        """

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

    def calc_pucker_amplitude(self) -> float:
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

    Attributes:
        atoms: The atoms in the linkage.
        start_ring: The start ring, containing atoms[0].
        end_ring: The end ring, containing atoms[-1].
    """

    atoms: list[Atom]
    start_ring: CarbRing
    end_ring: CarbRing

    def calc_angles(self) -> FloatArray:
        """Calculate the dihedral angles associated with the linkage."""

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


def find_rings(structure: Structure, max_size: int) -> list[CarbRing]:
    """Find all rings in structure, with maximum size max_size."""

    rings = [
        CarbRing.from_ring(ring)
        for ring in structure.rings(cross_residue=False, all_size_threshold=max_size)
    ]
    # print(f"RINGS: {len(rings)}")
    return rings


@time
def find_linkages(rings: list[CarbRing], max_len: int) -> list[CarbLinkage]:
    """Find all linkages between the rings, with maximum length max_len."""

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

            for linkage in dfs_paths(
                get_neighbors,
                start_atom,
                visited,
                max_len=max_len,
            ):
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


def paperchain_colormap(ring: CarbRing) -> FloatArray:
    """
    Calculate the color for a ring, using the PaperChain algorithm.

    Returns:
        The calculated color as an RGB [0, 1] float array.
    """

    pucker = ring.calc_pucker_amplitude()

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


def dihedral_norm_colormap(linkage: CarbLinkage) -> FloatArray:
    angles = linkage.calc_angles()
    n = angles.shape[0]
    if n == 0:
        return np.zeros(3, dtype=np.float32)

    v = np.linalg.norm(angles)
    rgb = np.empty(3, dtype=np.float32)

    v /= np.sqrt(n) * 180
    rgb[0] = 1
    rgb[1] = 0.7 * (1 - v)
    rgb[2] = 0.7 * (1 - v)

    return rgb


def dihedral_colormap(linkage: CarbLinkage) -> FloatArray:
    angles = linkage.calc_angles()

    n = angles.shape[0]
    if n < 2 or n > 3:
        print(f"warning: linkage {linkage} has {n} angles")
        return np.zeros(3, dtype=np.float32)

    link_type = linkage.start_ring.residue.name[:1].lower()
    if link_type not in ("a", "b"):
        print(f"warning: linkage {linkage} has an unknown type {link_type!r}")
        return np.zeros(3, dtype=np.float32)

    if n == 2:
        phi, psi = angles

        if link_type == "a":
            nx = gaussian(phi, 1, -40, 40, 2)
        else:
            nx = gaussian(phi, 1, 40, 40, 2)
        ny = gaussian(psi, 1, 0, 60, 2)
        v = 1 - nx * ny
    else:
        phi, psi, omega = angles

        if link_type == "a":
            nx = gaussian(phi, 1, -40, 40, 2)
        else:
            nx = gaussian(phi, 1, 40, 40, 2)

        ny = 1 - (
            (1 - gaussian(psi, 1, 60, 40, 2))
            * (1 - gaussian(psi, 1, -60, 40, 2))
            * (1 - gaussian(abs(psi), 1, 180, 40, 2))
        )
        nz = 1 - (
            (1 - gaussian(omega, 1, 60, 40, 2))
            * (1 - gaussian(omega, 1, -60, 40, 2))
            * (1 - gaussian(abs(omega), 1, 180, 40, 2))
        )
        v = 1 - nx * ny * nz

    # if v > 0.1:
    #     print(
    #         f"{linkage.atoms[0]}->{linkage.atoms[-1]}:\n"
    #         f"type={link_type} angles={angles}"
    #     )
    #     print(f"v={v}\n")

    rgb = np.empty(3, dtype=np.float32)

    rgb[0] = 1
    rgb[1] = 0.7 * (1 - v)
    rgb[2] = 0.7 * (1 - v)

    return rgb
