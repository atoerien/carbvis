from typing import cast

from chimerax.atomic import Atoms, Structure, Structures, get_triggers
from chimerax.atomic.changes import Changes
from chimerax.core.models import Model
from chimerax.core.session import Session

MODEL_STATE_VERSION = 1


class CarbVisModel(Model):
    """
    A model visualising a (carbohydrate) structure.
    """

    def __init__(
        self,
        session: Session,
        atoms: Atoms,
        name: str,
        *,
        update: bool,
    ):
        self.session: Session

        structures: Structures = atoms.unique_structures
        if len(structures) != 1:
            raise ValueError("Atoms must belong to one structure")
        structure = cast(Structure, structures[0])

        super().__init__(name, session)

        self.atoms = atoms
        self.structure = structure

        # change tracking stuff
        self._atom_count = len(atoms)
        self._bonds = atoms.bonds
        self._bond_count = len(atoms.bonds)

        self._auto_update_handler = None
        self.auto_update = update

    def delete(self):
        self.auto_update = False  # Remove auto update handler
        super().delete()

    def _clear_geometry(self):
        self.set_geometry(None, None, None)
        self.texture_coordinates = None

    @property
    def auto_update(self):
        return self._auto_update_handler is not None

    @auto_update.setter
    def auto_update(self, enable):
        h = self._auto_update_handler
        if enable and h is None:
            t = get_triggers()
            self._auto_update_handler = t.add_handler("changes", self._auto_update_cb)
        elif not enable and h is not None:
            t = get_triggers()
            t.remove_handler(h)
            self._auto_update_handler = None

    def _auto_update_cb(self, trigger_name, changes: Changes):
        if self.deleted or self.structure.deleted:
            return "delete handler"

        structure_changed = self._structure_changed(changes)
        coords_changed = self._coords_changed(changes)
        self._do_update(
            structure_changed=structure_changed,
            coords_changed=coords_changed,
        )

    def _do_update(
        self,
        *,
        structure_changed: bool,
        coords_changed: bool,
    ):
        """
        Called to update this model.

        Does nothing, to be overridden in subclasses.

        Args:
            structure_changed: If `True`, the atoms or bonds in the
                structure have changed.
            coords_changed: If `True`, the atom coordinates changed.
        """

    def update(self):
        """Update this model"""

        structure_changed = self._structure_changed()
        coords_changed = self._coords_changed()
        self._do_update(
            structure_changed=structure_changed,
            coords_changed=coords_changed,
        )

    def _structure_changed(self, changes: Changes | None = None):
        """Check whether the atoms or bonds in the structure have changed"""

        atoms = self.atoms
        bonds = atoms.bonds
        if len(atoms) != self._atom_count:
            # we never assign to self.atoms, so since it's "immutable"
            # we only need to check if atoms were deleted
            self._atom_count = len(atoms)
            self._bonds = bonds
            self._bond_count = len(bonds)
            return True
        if len(bonds) != self._bond_count or bonds != self._bonds:
            # bonds on the other hand could have changed in any way
            # check for length as well because deleted bonds will be
            # removed from self._bonds as well
            self._bonds = bonds
            self._bond_count = len(bonds)
            return True
        return False

    def _coords_changed(self, changes: Changes | None = None):
        """Check whether the structure's atom coordinates changed"""

        if changes is not None:
            if "active_coordset changed" in changes.structure_reasons():
                # playing a trajectory
                for s in changes.modified_structures():
                    if s == self.structure:
                        return True
            if "coord changed" in changes.atom_reasons():
                # atom coordinates changed through Atom or Atoms set_coord()
                if self.atoms.intersects(changes.modified_atoms()):
                    return True
            elif "coordset changed" in changes.coordset_reasons():
                # atom coordinates changed through CoordSet
                for cs in changes.modified_coordsets():
                    if cs.structure == self.structure:
                        return True
        else:
            # TODO: detect? hash the array? make sure not slow
            return True
        return False

    def take_snapshot(self, session: Session, flags: int):
        data = {
            "atoms": self.atoms,
            "name": self.name,
            "update": self.auto_update,
        }
        data["model state"] = Model.take_snapshot(self, session, flags)
        data["version"] = MODEL_STATE_VERSION
        return data

    @classmethod
    def restore_snapshot(cls, session: Session, data: dict):
        ret = CarbVisModel(
            session,
            data["atoms"],
            data["name"],
            update=data["update"],
        )
        ret.set_state_from_snapshot(session, data)
        return ret

    def set_state_from_snapshot(self, session, data):
        Model.set_state_from_snapshot(self, session, data["model state"])
