# テスト用のデータモヂュール
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from dataset import NiftiDataset


class DataModuleForTest(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size=2):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size

    def setup(self, stage=None):
        print("Preparing test data...")
        image_folder = self.dataset_path / "images"
        mask_folder = self.dataset_path / "masks"

        # データセット全体をテスト用にロード
        self.test_dataset = NiftiDataset(image_folder, mask_folder)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
