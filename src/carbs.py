from dataclasses import dataclass, field
from itertools import pairwise
from typing import Self, cast

import numpy as np
from chimerax.atomic import Atom, Atoms, Bond, Bonds, Element, Residue, Residues, Ring
from chimerax.core.commands import run
from chimerax.core.session import Session
from chimerax.core.state import State
from chimerax.geometry import dihedral
from chimerax.graphics import Pick

from ._carbs import (  # pyright: ignore[reportMissingModuleSource]
    paperchain_colormap as paperchain_colormap,
)
from ._carbs import (  # pyright: ignore[reportMissingModuleSource]
    ring_calc_pucker_amplitude_py,
    ring_get_centroid_and_normal_py,
    ring_get_centroid_py,
    ring_get_frame_py,
)
from .utils import DoubleArray, FloatArray, Frame, dfs_paths, gaussian


@dataclass
class CarbRing(State):
    """
    A ring of atoms connected to each other in a loop.

    Attributes:
        atoms: The atoms in the ring.
        bonds: The bonds between the atoms.
        residue: The residue the atoms are part of.
        orientation: Indicates if the order of the atoms corresponds
            to the orientation (handedness) of the ring.
    """

    atoms: Atoms
    bonds: Bonds
    residue: Residue
    orientated: bool = field(default=False)

    @classmethod
    def from_ring(cls, ring: Ring) -> Self:
        """
        Create an instance from a Ring object.

        The ring must not cross a residue boundary.
        """

        atoms: Atoms = ring.ordered_atoms
        bonds: Bonds = ring.ordered_bonds

        # roll the atoms list for a consistent ring ordering
        roll_shift = -np.argmin([a.name for a in atoms])
        if roll_shift != 0:
            atoms = Atoms(np.roll(atoms.pointers, roll_shift))
            bonds = Bonds(np.roll(bonds.pointers, roll_shift))

        residues: Residues = atoms.unique_residues
        if len(residues) != 1:
            raise ValueError("Ring crosses residue boundary")

        ret = cls(atoms, bonds, residues[0])

        ret.orientate()

        return ret

    def __str__(self) -> str:
        res = str(self.residue)
        atoms = ",".join(a.name for a in self.atoms)
        return f"{res} {atoms}"

    @property
    def atomspec(self) -> str:
        res = self.residue.atomspec
        atoms = ",".join(a.name for a in self.atoms)
        return f"{res}@{atoms}"

    @property
    def session(self) -> Session:
        return self.residue.session

    @property
    def selected(self) -> bool:
        if self.atoms.num_selected != len(self.atoms):
            return False
        if self.bonds.num_selected != len(self.bonds):
            return False
        return True

    @selected.setter
    def selected(self, sel: bool):
        self.atoms.selected = sel
        self.bonds.selected = sel

    def orientate(self):
        """
        Orientate the ring, flipping the atom list such that the
        order corresponds to the handedness of the ring.

        Returns:
            Whether the ring was able to be oriented or not.
        """
        if self.orientated:
            return

        oxygen = -1

        # Find an oxygen (or something with two bonds)
        for i, atom in enumerate(self.atoms):
            element: Element = atom.element
            if element.number == 8 or atom.num_bonds == 2:
                oxygen = i
                break

        if oxygen == -1:
            log = self.session.logger
            log.warning(
                f"ring {self} does not contain an oxygen atom, cannot orientate"
            )
            return

        # find atoms before and after oxygen (taking into account wrapping)
        atoms = self.atoms
        atom_before_O = cast(Atom, atoms[oxygen - 1])
        atom_after_O = cast(Atom, atoms[(oxygen + 1) % len(atoms)])

        # ensure C1 carbon is after the oxygen
        # leave unorientated if the C1 carbon can't be found
        if atom_before_O.name in ("C1", "C1'", "C_1", "C2", "C2'", "C_2"):
            # reverse atom list, while keeping the first atom in the same place
            ptrs = self.atoms.pointers.copy()
            ptrs[1:] = np.flip(ptrs[1:])
            self.atoms = Atoms(ptrs)
            self.orientated = True
        elif atom_after_O.name in ("C1", "C1'", "C_1", "C2", "C2'", "C_2"):
            self.orientated = True
        else:
            log = self.session.logger
            log.warning(f"ring {self} does not contain a C1 carbon, cannot orientate")

    def get_centroid(self) -> DoubleArray:
        """Calculate the ring centroid."""
        return ring_get_centroid_py(self)

    def get_centroid_and_normal(self) -> tuple[DoubleArray, DoubleArray]:
        """
        Calculate the ring centroid and normal vector.

        The returned normal vector points in the direction of the
        "top" of the ring (a left-handed normal).

        Returns:
            A tuple (centroid, normal).
        """
        return ring_get_centroid_and_normal_py(self)

    def get_frame(self) -> Frame:
        """
        Return a Frame corresponding to the plane of the ring.

        The forward vector of the frame points towards the first atom,
        and the up vector is the ring normal.
        """
        return ring_get_frame_py(self)

    def calc_pucker_amplitude(self) -> float:
        """
        Calculate the Cremer-Pople puckering amplitude for this ring.
        """
        return ring_calc_pucker_amplitude_py(self)

    def take_snapshot(self, session, flags):
        return {
            "atoms": self.atoms,
            "bonds": self.bonds,
            "residue": self.residue,
            "orientated": self.orientated,
        }

    @classmethod
    def restore_snapshot(cls, session, data):
        return cls(**data)


class PickedRing(Pick):
    def __init__(self, ring: CarbRing, distance):
        super().__init__(distance)
        self.ring = ring

    def description(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return str(self.ring)

    @property
    def residue(self) -> Residue:
        return self.ring.residue

    def select(self, mode="add"):
        ring = self.ring
        session = ring.residue.session
        if mode == "add" or (mode == "toggle" and not ring.selected):
            run(session, f"select add {ring.atomspec}")
        else:
            run(session, f"select subtract {ring.atomspec}")

    def specifier(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.ring.atomspec


class PickedRings(Pick):
    def __init__(self, rings: list[CarbRing]):
        super().__init__()
        self.rings = rings

    def description(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"{len(self.rings)} rings"

    @property
    def ring(self) -> CarbRing | None:
        if len(self.rings) == 1:
            return self.rings[0]
        return None

    def select(self, mode="add"):
        if mode == "add":
            for ring in self.rings:
                ring.selected = True
        elif mode == "subtract":
            for ring in self.rings:
                ring.selected = False
        elif mode == "toggle":
            for ring in self.rings:
                ring.selected = not ring.selected


@dataclass
class CarbLinkage(State):
    """
    A (C1->Cx) linkage between two rings.

    Attributes:
        atoms: The atoms in the linkage.
        bonds: The bonds in the linkage.
        start_ring: The start ring, containing atoms[0].
        end_ring: The end ring, containing atoms[-1].
    """

    atoms: Atoms
    bonds: Bonds
    start_ring: CarbRing
    end_ring: CarbRing

    def __str__(self) -> str:
        atoms = self.atoms
        a1 = cast(Atom, atoms[0])
        a2 = cast(Atom, atoms[-1])
        return f"{a1.string()} \N{RIGHTWARDS ARROW} {a2.string(relative_to=a1)}"

    @property
    def atomspec(self) -> str:
        if len(self.atoms) == 0:
            return ""
        ret = cast(Atom, self.atoms[0]).atomspec
        for a1, a2 in pairwise(self.atoms):
            s = a2.string(style="command", relative_to=a1)
            if s.startswith("@"):
                ret += f",{s[1:]}"
            else:
                ret += s
        return ret

    @property
    def session(self) -> Session:
        return self.start_ring.session

    @property
    def selected(self) -> bool:
        if self.atoms.num_selected != len(self.atoms):
            return False
        if self.bonds.num_selected != len(self.bonds):
            return False
        return True

    @selected.setter
    def selected(self, sel: bool):
        self.atoms.selected = sel
        self.bonds.selected = sel

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
                    log = self.session.logger
                    log.warning(
                        f"multiple non-ring atoms attached to {first_atom},"
                        f" dihedral angles for linkage {self} might be inaccurate"
                    )
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
                    log = self.session.logger
                    log.warning(
                        f"multiple non-ring atoms attached to {last_atom},"
                        f" dihedral angles for linkage {self} might be inaccurate"
                    )
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

    def take_snapshot(self, session, flags):
        return {
            "atoms": self.atoms,
            "bonds": self.bonds,
            "start_ring": self.start_ring,
            "end_ring": self.end_ring,
        }

    @classmethod
    def restore_snapshot(cls, session, data):
        return cls(**data)


class PickedLinkage(Pick):
    def __init__(self, linkage: CarbLinkage, distance):
        super().__init__(distance)
        self.linkage = linkage

    def description(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return str(self.linkage)

    def select(self, mode="add"):
        linkage = self.linkage
        session = linkage.start_ring.residue.session
        if mode == "add" or (mode == "toggle" and not linkage.selected):
            run(session, f"select add {linkage.atomspec}")
        else:
            run(session, f"select subtract {linkage.atomspec}")

    def specifier(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.linkage.atomspec


class PickedLinkages(Pick):
    def __init__(self, linkages: list[CarbLinkage]):
        super().__init__()
        self.linkages = linkages

    def description(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"{len(self.linkages)} linkages"

    @property
    def linkage(self) -> CarbLinkage | None:
        if len(self.linkages) == 1:
            return self.linkages[0]
        return None

    def select(self, mode="add"):
        if mode == "add":
            for link in self.linkages:
                link.selected = True
        elif mode == "subtract":
            for link in self.linkages:
                link.selected = False
        elif mode == "toggle":
            for link in self.linkages:
                link.selected = not link.selected


@line_profile
def find_rings(atoms: Atoms, max_size: int) -> list[CarbRing]:
    """Find all rings in atoms, with maximum size max_size."""

    rings = [
        CarbRing.from_ring(ring)
        for structure, structure_atoms in atoms.by_structure
        for ring in structure.rings(cross_residue=False, all_size_threshold=max_size)
        # all ring atoms must be present in selected atoms
        if len(ring.atoms - structure_atoms) == 0
    ]
    # print(f"RINGS: {len(rings)}")
    return rings


@line_profile
def find_linkages(rings: list[CarbRing], max_len: int) -> list[CarbLinkage]:
    """Find all linkages between the rings, with maximum length max_len."""

    atom_to_ring: dict[Atom, CarbRing] = {}
    multi_ring_atoms: set[Atom] = set()

    for start_ring in rings:
        for atom in start_ring.atoms:
            if atom in atom_to_ring:
                multi_ring_atoms.add(atom)
            else:
                atom_to_ring[atom] = start_ring

    def get_neighbors(node: tuple[Atom, Bond | None]):
        atom, bond = node
        if bond is None:
            # the first atom has bond None, only allow edges out of the ring
            ret = [
                (a, b)
                for a, b in zip(atom.neighbors, atom.bonds)
                if a not in atom_to_ring
            ]
        elif atom in atom_to_ring:
            # otherwise if we're in a ring it's an endpoint
            ret = []
        else:
            ret = [(a, b) for a, b in zip(atom.neighbors, atom.bonds)]
        return ret

    linkages: list[CarbLinkage] = []

    # keep the same visited set through all the dfs_paths calls
    # start_atom then stays in the set, preventing duplicate
    # forward and reverse paths
    visited: set[int] = set()

    for start_ring in rings:
        if not start_ring.orientated:
            continue

        for start_atom in start_ring.atoms:
            if start_atom in multi_ring_atoms:
                continue

            id_start_atom = id(start_atom)
            if id_start_atom in visited:
                continue
            visited.add(id_start_atom)

            for atom_path in dfs_paths(
                get_neighbors,
                (start_atom, None),
                visited,
                max_len=max_len,
            ):
                end_atom, _ = atom_path[-1]
                if start_atom is end_atom:
                    continue

                if end_atom in multi_ring_atoms:
                    continue

                if end_atom not in atom_to_ring:
                    continue
                end_ring = atom_to_ring[end_atom]

                atoms, bonds = map(list, zip(*atom_path))
                bonds = bonds[1:]  # drop the first None

                # enforce ordering, start_ring should be the C1 carbon
                # also if no C1 involved at all, start_ring can be the C2 carbon
                if start_atom.name in ("C1", "C1'", "C_1"):
                    sring = start_ring
                    ering = end_ring
                elif end_atom.name in ("C1", "C1'", "C_1"):
                    # flip
                    sring = end_ring
                    ering = start_ring
                    atoms.reverse()
                    bonds.reverse()
                elif start_atom.name in ("C2", "C2'", "C_2"):
                    sring = start_ring
                    ering = end_ring
                elif end_atom.name in ("C2", "C2'", "C_2"):
                    # flip
                    sring = end_ring
                    ering = start_ring
                    atoms.reverse()
                    bonds.reverse()
                else:
                    log = start_atom.session.logger
                    log.warning(
                        f"linkage {start_atom}->{end_atom} is not a"
                        " (C1->Cx) linkage, cannot enforce direction"
                    )
                    sring = start_ring
                    ering = end_ring

                linkages.append(CarbLinkage(Atoms(atoms), Bonds(bonds), sring, ering))

    # print(f"LINKAGES: {len(linkages)}")
    return linkages


@line_profile
def dihedral_norm_colormap(linkage: CarbLinkage) -> FloatArray:
    angles = linkage.calc_angles()
    n = angles.shape[0]
    if n == 0:
        return np.zeros(4, dtype=np.float32)

    v = np.linalg.norm(angles)
    v /= np.sqrt(n) * 180

    ret = np.empty(4, dtype=np.float32)
    ret[0] = 1
    ret[1] = 0.7 * (1 - v)
    ret[2] = 0.7 * (1 - v)
    ret[3] = 1
    return ret


@line_profile
def dihedral_colormap(linkage: CarbLinkage) -> FloatArray:
    angles = linkage.calc_angles()

    n = angles.shape[0]
    if n < 2 or n > 3:
        log = linkage.session.logger
        log.warning(f"linkage {linkage} has {n} angles, cannot calculate color")
        return np.ones(4, dtype=np.float32)

    link_type = linkage.start_ring.residue.name[:1].lower()
    if link_type not in ("a", "b"):
        log = linkage.session.logger
        log.warning(
            f"linkage {linkage} has an unknown type {link_type!r}, cannot calculate color"
        )
        return np.ones(4, dtype=np.float32)

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

    ret = np.empty(4, dtype=np.float32)

    ret[0] = 1
    ret[1] = 0.7 * (1 - v)
    ret[2] = 0.7 * (1 - v)
    ret[3] = 1

    return ret
