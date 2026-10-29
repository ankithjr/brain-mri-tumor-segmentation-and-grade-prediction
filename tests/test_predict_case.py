"""CPU checks; no model checkpoint or GPU required."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("predict_case", ROOT / "scripts/predict_case.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = []
        rng = np.random.default_rng(42)
        for channel in pipeline.CHANNELS:
            volume = rng.uniform(1, 100, (40, 42, 38)).astype(np.float32)
            volume[:2] = 0
            image = nib.Nifti1Image(volume, np.eye(4))
            image.header.set_xyzt_units("mm")
            path = self.root / f"{channel}.nii.gz"
            nib.save(image, path)
            self.paths.append(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_matches_executed_crop_script(self):
        # Run the actual archived preprocessing source on a two-case synthetic
        # fixture. Only its dataset root and expected cohort size are substituted.
        images, volumes = pipeline.load_images(self.paths)
        mask_folder = self.root / "nnUNet_results/Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
        mask_folder.mkdir(parents=True)
        (self.root / "data").mkdir()
        for channel, path in zip(pipeline.CHANNELS, self.paths):
            (self.root / "data" / path.name).symlink_to(path)
        rows, masks = [], []
        for name, empty in (("boundary", False), ("empty", True)):
            mask = np.zeros(volumes[0].shape, dtype=np.uint8)
            if not empty:
                mask[1:4, 20:23, 35:38] = 3
            nib.save(nib.Nifti1Image(mask, images[0].affine), mask_folder / f"{name}.nii.gz")
            rows.append({"exam_id": name, "grade": 2, **{c: p.name for c, p in zip(pipeline.CHANNELS, self.paths)}})
            masks.append(mask)
        split_folder = self.root / "results/splits"
        split_folder.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(split_folder / "validation.csv", index=False)
        source = (ROOT / "scripts/prepare_predicted_val_crops.py").read_text()
        source = source.replace('Path("/scratch/markov3/users/arj94/brain-glioma")', f"Path({str(self.root)!r})")
        source = source.replace("assert len(frame) == 74", "assert len(frame) == 2")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "archived_preprocessing", "exec"), {"__name__": "__main__"})
        for row, mask in zip(rows, masks):
            actual, bounds, fallback = pipeline.prepare_crop(volumes, mask)
            with np.load(self.root / "preprocessed/classifier_predicted_crop_v1/validation" / f"{row['exam_id']}.npz") as sample:
                np.testing.assert_array_equal(actual, sample["image"])
            self.assertEqual(fallback, row["exam_id"] == "empty")
            self.assertEqual(actual.shape, (4, 96, 96, 64))
            self.assertTrue(all(0 <= a < b <= n for (a, b), n in zip(bounds, mask.shape)))

    def test_rejects_geometry_mismatch(self):
        image = nib.load(self.paths[-1])
        affine = image.affine.copy()
        affine[0, 3] += 2
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine, image.header), self.paths[-1])
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            pipeline.load_images(self.paths)

    def test_rejects_nonfinite_and_invalid_mask(self):
        _, volumes = pipeline.load_images(self.paths)
        bad = np.zeros(volumes[0].shape)
        bad[0, 0, 0] = 4
        with self.assertRaisesRegex(ValueError, "mask labels"):
            pipeline.prepare_crop(volumes, bad)
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "mask shape or values"):
            pipeline.prepare_crop(volumes, bad)


if __name__ == "__main__":
    unittest.main()
