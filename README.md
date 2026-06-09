# CarbVis

A [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) bundle implementing four carbohydrate-specific molecular visualizations:

- PaperChain, with PaperChain Texture variant
- Twister, with Twister Gum variant
- Strand, with Candy Cane variant
- Color by dihedral

## Requirements

- UCSF ChimeraX
- [Just](https://github.com/casey/just)

## Building

Run `just` to build the Python wheel in `build/`, and `just install` to install it into ChimeraX.

The Justfile searches for the `chimerax` command in `PATH`, if this is not present set the `CHIMERAX`
environment variable to the full path to the `chimerax` command.

## Usage

In the ChimeraX command-line, run `paperchain`, `twister`, `strand`, or `color bydihedral` respectively.

Prefix those commands with `help ` to open the command's documentation page inside ChimeraX.

## Attribution

The original PaperChain and Twister algorithms were ported from their implementations in [VMD](https://www.ks.uiuc.edu/Research/vmd/) by Simon Cross et al.
