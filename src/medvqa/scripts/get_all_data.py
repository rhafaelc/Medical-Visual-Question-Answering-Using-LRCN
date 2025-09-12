#!/usr/bin/env python3
from .get_vqa_rad_osf import main as vqa_main
from .get_slake_hf import main as slake_main


def main():
    rc1 = vqa_main()
    rc2 = slake_main()
    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
