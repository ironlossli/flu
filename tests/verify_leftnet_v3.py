import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra.models.ehc import build_ehc_leftnet_v3

def test_build():
    try:
        build_ehc_leftnet_v3({})
    except NotImplementedError as exc:
        print(f"SKIP: {exc}")
        return
    print("LeftNet v3 builder is unexpectedly enabled.")

if __name__ == "__main__":
    test_build()
