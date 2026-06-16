import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tests.test_pahalgram_descendant_clustering import test_pahalgram_descendant_clustering

if __name__ == "__main__":
    test_pahalgram_descendant_clustering()
    print("Test passed successfully!")
