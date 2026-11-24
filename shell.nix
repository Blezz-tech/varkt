{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "varkt-gravity-assist";

  packages = with pkgs; [
    (python3.withPackages (ps: with ps; [
      numpy
      matplotlib
      typst
      pymupdf
    ]))
    typst
    ffmpeg
  ];

  shellHook = ''
    export MPLBACKEND=Agg
    echo "Окружение VARKT готово."
    echo "  python3 src/sim.py                        # расчёт модели и 5 графиков"
    echo "  python3 src/jpl_horizons.py --no-network  # верификация по «Вояджеру-2»"
    echo "  python3 src/anim.py                       # генерация видео out/mission.mp4"
    echo "  cd report && python3 build.py             # сборка отчёта report/report.pdf"
  '';
}
