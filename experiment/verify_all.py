"""Verification script for SG-DETR + DQ-CGP Migration.

Runs all acceptance tests and prints a comprehensive audit report.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("=" * 80)
    print("       SG-DETR + DQ-CGP Migration: Acceptance Criteria Verification")
    print("=" * 80)

    loader = unittest.TestLoader()
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(test_dir, pattern="test_*.py", top_level_dir=project_root)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 80)
    if result.wasSuccessful():
        print(f"✅ ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        print("SG-DETR + DQ-CGP migration complies with all requirements in shuoming.md.")
        sys.exit(0)
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors detected out of {result.testsRun} tests.")
        sys.exit(1)

if __name__ == "__main__":
    main()
