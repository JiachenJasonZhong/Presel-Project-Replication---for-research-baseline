"""验证环境安装是否正确"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies():
    missing = []

    try:
        import numpy
        print(f"numpy: {numpy.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        import torch
        print(f"torch: {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        from sklearn.cluster import KMeans
        print("scikit-learn: ok")
    except ImportError:
        missing.append("scikit-learn")

    try:
        from presel import PreSel
        from presel.data import create_dummy_dataset
        print("presel: ok")
    except ImportError as e:
        missing.append(f"presel ({e})")

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\nAll dependencies installed.")
    return True


if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)