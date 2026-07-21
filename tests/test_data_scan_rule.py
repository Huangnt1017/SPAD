"""正式三页窗口文件名与数据扫描规则测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.data import (
    _collect_point_files,
    _is_point_file,
    discover_spad_classification_samples,
    parse_formal_window_filename,
)


class FormalWindowFilenameTests(unittest.TestCase):
    """正式窗口命名解析测试。"""

    def test_valid_filename_returns_acquisition_and_window(self) -> None:
        file_name = "2025-04-30_18-47-35_Delay-0_Width-200-1-3.txt"
        parsed = parse_formal_window_filename(file_name)
        self.assertEqual(
            parsed,
            ("2025-04-30_18-47-35_Delay-0_Width-200", 1, 3),
        )
        self.assertTrue(_is_point_file(file_name))

    def test_only_i_to_i_plus_two_is_accepted(self) -> None:
        self.assertTrue(
            _is_point_file(
                "2025-04-30_18-47-35_Delay-0_Width-200-98-100.txt"
            )
        )
        self.assertFalse(
            _is_point_file(
                "2025-04-30_18-47-35_Delay-0_Width-200-1-2.txt"
            )
        )
        self.assertFalse(
            _is_point_file(
                "2025-04-30_18-47-35_Delay-0_Width-200-1-4.txt"
            )
        )
        self.assertFalse(
            _is_point_file(
                "2025-04-30_18-47-35_Delay-0_Width-200-0-2.txt"
            )
        )

    def test_auxiliary_and_malformed_files_are_rejected(self) -> None:
        invalid_names = (
            "A.txt",
            "2025-04-30_18-47-35_Delay-0_Width-200-1-3.json",
            "2025-04-30_18-47-35_Delay-0_Width-200-1-3.npy",
            "2025-04-30_18-47-35_Delay-0_Width-201-1-3.txt",
            "2025-04-30_18-47-35_Delay-1_Width-200-1-3.txt",
            "2025-04-30_18-47-35_Delay-0_Width-200-1-3_hmc.txt",
            "2025-04-30_18-47_Delay-0_Width-200-1-3.txt",
            "2025-13-30_18-47-35_Delay-0_Width-200-1-3.txt",
            "2025-04-30_25-47-35_Delay-0_Width-200-1-3.txt",
        )
        for file_name in invalid_names:
            with self.subTest(file_name=file_name):
                self.assertIsNone(parse_formal_window_filename(file_name))
                self.assertFalse(_is_point_file(file_name))


class FormalWindowDirectoryScanTests(unittest.TestCase):
    """目录扫描只保留正式窗口文件。"""

    def test_collect_and_discover_ignore_auxiliary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            class_a = root / "A"
            class_b = root / "B"
            class_a.mkdir()
            class_b.mkdir()

            valid_a = (
                class_a
                / "2025-04-30_18-47-35_Delay-0_Width-200-1-3.txt"
            )
            valid_b = (
                class_b
                / "2025-04-30_18-48-37_Delay-0_Width-200-2-4.txt"
            )
            invalid_files = (
                class_a / "A.txt",
                class_a
                / "2025-04-30_18-47-35_Delay-0_Width-200-1-3.json",
                class_b
                / "2025-04-30_18-48-37_Delay-0_Width-200-2-4_hmc.txt",
            )

            valid_a.write_text("1,1,1,1\n", encoding="utf-8")
            valid_b.write_text("1,1,1,1\n", encoding="utf-8")
            for path in invalid_files:
                path.write_text("1,1,1,1\n", encoding="utf-8")

            collected = _collect_point_files(str(root))
            self.assertEqual(
                collected,
                sorted((str(valid_a), str(valid_b))),
            )

            labeled, unlabeled = discover_spad_classification_samples(
                str(root)
            )
            self.assertEqual(len(labeled), 2)
            self.assertEqual(len(unlabeled), 0)
            self.assertEqual(
                sorted(sample["label"] for sample in labeled),
                ["A", "B"],
            )


if __name__ == "__main__":
    unittest.main()
