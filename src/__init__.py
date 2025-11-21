from chimerax.core.logger import Logger
from chimerax.core.toolshed import BundleAPI, BundleInfo, CommandInfo


class _CarbVisAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi: BundleInfo, ci: CommandInfo, logger: Logger):  # pyright: ignore[reportIncompatibleMethodOverride]
        from . import cmd

        name = ci.name.replace(" ", "_")
        try:
            fn: cmd.CmdFunc = getattr(cmd, name)
            desc = fn.desc
        except AttributeError:
            raise ValueError(f"unknown command: {ci.name}")

        if desc.synopsis is None:
            desc.synopsis = ci.synopsis

        from chimerax.core.commands import register

        register(ci.name, desc, fn)

    @staticmethod
    def get_class(name: str):  # pyright: ignore[reportIncompatibleMethodOverride]
        # needed for model save/restore

        if name == "CarbVisModel":
            from .model import CarbVisModel

            return CarbVisModel
        elif name == "PaperChainModel":
            from .paperchain import PaperChainModel

            return PaperChainModel
        elif name == "StrandModel":
            from .strand import StrandModel

            return StrandModel
        elif name == "TwisterModel":
            from .twister import TwisterModel

            return TwisterModel
        elif name == "CarbRing":
            from .carbs import CarbRing

            return CarbRing
        elif name == "CarbLinkage":
            from .carbs import CarbLinkage

            return CarbLinkage
        else:
            return None


bundle_api = _CarbVisAPI()
