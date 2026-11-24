"""Сборка отчёта: report.typ -> report.pdf.

Запуск:  python3 build.py
Требуется: pip install typst
Шрифты берутся из ../fonts (Times New Roman для текста, STIX Two Math для формул).
"""
import os
import sys

import typst

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)              # корень репозитория
FONTS = os.path.join(ROOT, "fonts")

os.chdir(HERE)


def run() -> None:
    font_paths = [FONTS] if os.path.isdir(FONTS) else []
    typst.compile(
        "report.typ",
        output="report.pdf",
        font_paths=font_paths,
        root=ROOT,
        pretty=False,
    )
    size = os.path.getsize(os.path.join(HERE, "report.pdf"))
    print(f"OK: report.pdf ({size / 1024:.0f} КБ)")
    if not font_paths:
        print("ВНИМАНИЕ: каталог fonts/ не найден — возможна подмена шрифта.", file=sys.stderr)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # noqa: BLE001
        print("ERR:", e)
        sys.exit(1)
