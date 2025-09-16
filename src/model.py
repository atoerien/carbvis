import numpy as np
from chimerax.atomic import Structure, get_triggers
from chimerax.atomic.changes import Changes
from chimerax.core.models import Model
from chimerax.core.session import Session


class CarbVisModel(Model):
    """
    A model visualising a (carbohydrate) structure.
    """

    def __init__(
        self,
        session: Session,
        structure: Structure,
        name: str | None = None,
        *,
        update: bool,
    ):
        self.session: Session

        if name is None:
            name = f"{structure.name} CarbVis"
        super().__init__(name, session)

        # TODO: what does this do?
        # self.selection_coupled = atoms.unique_structures

        self.structure = structure
        self._atom_count = structure.num_atoms  # Used to check if atoms deleted
        self._bond_count = structure.num_bonds  # Used to check if bonds deleted

        self._auto_update_handler = None
        self.auto_update = update

    def delete(self):
        self.auto_update = False  # Remove auto update handler
        super().delete()

    def _clear_geometry(self):
        va = na = np.empty((0, 3), np.float32)
        ta = np.empty((0, 3), np.int32)
        self.set_geometry(va, na, ta)
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
            if self.vertices is not None:
                self._recalculate_graphics()
        elif not enable and h:
            t = get_triggers()
            t.remove_handler(h)
            self._auto_update_handler = None

    def _auto_update_cb(self, trigger_name, changes: Changes):
        if self.deleted:
            return "delete handler"
        self._do_auto_update(changes)

    def _do_auto_update(self, changes: Changes):
        # do nothing, override in subclasses
        pass

    def _structure_changed(self, changes: Changes):
        """Check whether the atoms or bonds in the structure changed"""

        structure = self.structure
        if structure.num_atoms != self._atom_count:
            self._atom_count = structure.num_atoms
            # bond count might also have changed
            self._bond_count = structure.num_bonds
            return True
        if structure.num_bonds != self._bond_count:
            self._bond_count = structure.num_bonds
            return True
        if "active_coordset changed" in changes.structure_reasons():
            # Active coord set index changed.  Playing a trajectory.
            for s in changes.modified_structures():
                if s == self.structure:
                    return True
        if "coordset changed" in changes.coordset_reasons():
            # Atom coordinates changed through CoordSet object.
            for cs in changes.modified_coordsets():
                if cs.structure == self.structure:
                    return True
        return False

    def _recalculate_graphics(self):
        if self.structure.num_atoms == 0:
            self.session.models.close([self])
        else:
            self._clear_geometry()
            self.calculate_graphics()

    def _calc_graphics(self):
        # do nothing, override in subclasses
        pass

    def calculate_graphics(self):
        """Recalculate the geometry and color if parameters have been changed."""

        if self.vertices is not None and len(self.vertices) > 0:
            # Geometry already computed
            return

        self._calc_graphics()

    # TODO: make selection select the whole chain/ring/etc
    # TODO: make hide/show work
    # TODO: make save/restore work
