from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.pol_sources import SOURCE_FILES, SOURCE_NAMES, load_pol_source_inventory  # noqa: E402
import sim.polsim as polsim  # noqa: E402


SIM_DIR = REPO_ROOT / "sim"


class PolSourceInventoryTests(unittest.TestCase):
    def test_named_inventory_shapes_order_and_metadata(self) -> None:
        inventory = load_pol_source_inventory(SIM_DIR)

        self.assertEqual(inventory.source_names, list(SOURCE_NAMES))
        self.assertEqual(inventory.source_maps.shape, (7, 40, 40))
        self.assertEqual(inventory.source_matrix.shape, (1600, 7))
        self.assertFalse(hasattr(inventory, "aggregate_source"))
        self.assertEqual(inventory.raw_metadata["normalization"], "per_source_cropped_p99")
        self.assertEqual(inventory.raw_metadata["crop"]["axis0"], (21, 61, None))
        self.assertEqual(inventory.raw_metadata["crop"]["axis1"], (16, 56, None))
        self.assertEqual(
            inventory.source_activity_defaults["traffic_time_slices"],
            ["traffic_00", "traffic_06", "traffic_12", "traffic_18"],
        )

        for name in SOURCE_NAMES:
            self.assertIn(name, inventory.raw_metadata["source_files"])
            self.assertIn(name, inventory.raw_metadata["scale_by_source"])
            self.assertTrue(inventory.raw_metadata["source_files"][name].endswith(SOURCE_FILES[name]))

    def test_per_source_p99_scaling_and_zero_source_handling(self) -> None:
        inventory = load_pol_source_inventory(SIM_DIR)

        for i, name in enumerate(inventory.source_names):
            source_map = inventory.source_maps[i]
            if name == "traffic_06":
                self.assertTrue(np.array_equal(source_map, np.zeros_like(source_map)))
                self.assertIn(name, inventory.raw_metadata["all_zero_sources"])
                self.assertEqual(inventory.raw_metadata["scale_by_source"][name], 1.0)
            else:
                self.assertGreater(inventory.raw_metadata["scale_by_source"][name], 0.0)
                self.assertGreater(np.count_nonzero(source_map), 0)
                self.assertAlmostEqual(float(np.percentile(source_map, 99)), 1.0, places=5)

    def test_make_grid_inventory_is_opt_in_and_non_aggregate(self) -> None:
        empty_grid = polsim.make_grid(
            Nx=40,
            Ny=40,
            src_dir=str(SIM_DIR),
            device="cpu",
            dtype=torch.float32,
        )
        self.assertIsNone(empty_grid.S_known)
        self.assertIsNone(empty_grid.source_maps)

        grid = polsim.make_grid(
            Nx=40,
            Ny=40,
            src_dir=str(SIM_DIR),
            device="cpu",
            dtype=torch.float32,
            load_inventory=True,
        )

        self.assertIsNone(grid.S_known)
        self.assertEqual(grid.source_names, list(SOURCE_NAMES))
        self.assertEqual(tuple(grid.source_maps.shape), (7, 40, 40))
        self.assertEqual(tuple(grid.source_matrix.shape), (1600, 7))
        self.assertEqual(grid.source_metadata["normalization"], "per_source_cropped_p99")


if __name__ == "__main__":
    unittest.main()
