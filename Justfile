set unstable

export CHIMERAX := env('CHIMERAX', '') || which('ChimeraX') || which('chimerax')
run := CHIMERAX + " --nogui --exit --cmd"

wheel:
    {{run}} 'devel build . exit true'

install: wheel
    {{run}} 'toolshed uninstall CarbVis'
    whl="$(ls dist | head -n 1)"; if [ -n "$whl" ]; then {{run}} "toolshed install dist/$whl"; fi

install-rc: wheel
    ./rc.sh 'toolshed uninstall CarbVis'
    whl="$(ls dist | head -n 1)"; if [ -n "$whl" ]; then ./rc.sh "toolshed install $PWD/dist/$whl"; fi

clean:
    #!/usr/bin/env bash
    set -euxo pipefail
    if command -v "$CHIMERAX" &> /dev/null; then
        {{run}} 'devel clean . exit true'
    else
        rm -rf build dist *.egg-info src/__pycache__
    fi
