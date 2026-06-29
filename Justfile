export CHIMERAX := env("CHIMERAX", "ChimeraX")
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
    rm -rf build dist *.egg-info src/__pycache__
    for f in src_cy/chimerax/carbvis/*.pyx; do rm -f "${f%.pyx}"{.cpp,.html}; done
