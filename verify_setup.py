"""
SafeStride — Environment Verification Script
Imports every required package, prints version, reports PASS/FAIL.
"""
import importlib
import sys

PACKAGES = {
    "osmnx":          "osmnx",
    "networkx":       "networkx",
    "fastapi":        "fastapi",
    "uvicorn":        "uvicorn",
    "psycopg2":       "psycopg2",
    "geoalchemy2":    "geoalchemy2",
    "pandas":         "pandas",
    "numpy":          "numpy",
    "sklearn":        "sklearn",
    "torch":          "torch",
    "matplotlib":     "matplotlib",
    "dotenv":         "dotenv",
    "sqlalchemy":     "sqlalchemy",
    "shapely":        "shapely",
    "geopandas":      "geopandas",
}

def check_package(display_name: str, import_name: str) -> bool:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {display_name:20s} v{version}")
        return True
    except ImportError as e:
        print(f"  ❌ {display_name:20s} — {e}")
        return False

def main():
    print("=" * 50)
    print("  SafeStride Environment Verification")
    print("=" * 50)
    print(f"\n  Python {sys.version}\n")

    passed, failed = 0, 0
    for display_name, import_name in PACKAGES.items():
        if check_package(display_name, import_name):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All packages installed correctly!")
    else:
        print("  ⚠️  Some packages failed — review errors above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
