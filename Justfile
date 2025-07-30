chimerax := env('CHIMERAX', `which chimerax`)
run := chimerax + " --nogui --exit --cmd"

wheel:
    {{run}} 'devel build . exit true'

install: wheel
    {{run}} 'toolshed uninstall CarbVis'
    whl="$(ls dist | head -n 1)"; if [ -n "$whl" ]; then {{run}} "toolshed install dist/$whl"; fi

install-rc: wheel
    ./rc.sh 'toolshed uninstall CarbVis'
    whl="$(ls dist | head -n 1)"; if [ -n "$whl" ]; then ./rc.sh "toolshed install $PWD/dist/$whl"; fi

clean:
    #! bash
    if command -v chimerax &> /dev/null; then
        {{run}} 'devel clean . exit true'
    else
        rm -rf build dist *.egg-info src/__pycache__
    fi
