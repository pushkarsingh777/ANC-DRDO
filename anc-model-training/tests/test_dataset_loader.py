"""
Unit tests for SpeechEnhancementDataset & DataLoader (FR-2).
"""
import os
import sys
import unittest
# pyrefly: ignore [missing-import]
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import SpeechEnhancementDataset, get_dataloaders


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.manifest_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "anc-dataset-pipeline", "data", "manifests"))
        self.train_csv = os.path.join(self.manifest_dir, "mixtures_train.csv")

    def test_train_dataset_crop_shape(self):
        if not os.path.exists(self.train_csv):
            self.skipTest(f"Manifest not found at {self.train_csv}")
            
        ds = SpeechEnhancementDataset(
            manifest_path=self.train_csv,
            sample_rate=16000,
            segment_length_s=2.0,
            is_train=True,
        )
        self.assertGreater(len(ds), 0)
        
        sample = ds[0]
        self.assertEqual(sample["mixture"].shape, (32000,))
        self.assertEqual(sample["clean"].shape, (32000,))
        self.assertTrue(torch.all(torch.isfinite(sample["mixture"])))

    def test_dataloader_batching(self):
        if not os.path.exists(self.train_csv):
            self.skipTest(f"Manifest not found at {self.train_csv}")
            
        cfg = {
            "data": {
                "sample_rate": 16000,
                "segment_length_s": 1.5,
                "batch_size": 4,
                "eval_batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            }
        }
        train_loader, val_loader, test_loader = get_dataloaders(self.manifest_dir, cfg)
        
        batch = next(iter(train_loader))
        self.assertEqual(batch["mixture"].shape, (4, 24000))
        self.assertEqual(batch["clean"].shape, (4, 24000))
        self.assertEqual(len(batch["mixture_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
