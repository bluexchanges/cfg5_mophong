"""Small staged-cache tests: no dataset scan, GPU job, download, or long training."""
from pathlib import Path
from contextlib import closing
import sqlite3
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import sys
import torch
import numpy as np

from src.data.common import digest, read_json, write_json
from src.data.feature_cache import (cache_contract, cache_episode, check_episode,
                                   require_ready, FeatureACTDataset, FEATURE_ONLY, JPEG_ONLY)
from src.data.mp4_dataset import prepare_light
from src.data.sim_demos import inspect_episode, split_records, file_hash
from src.data.staged_mcr import (materialize_jpeg_episode, release_feature_episode,
                                validate_frozen_checkpoint, jpeg_storage_projection)
from tests import test_sim_finetune_data as fixtures
from tests.test_transfer_mcr import FakeEncoder


class StagedMCRTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SimFineTuneTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.cfg = self.fixture.cfg
        self.records = [inspect_episode(self.fixture.episode(i, gaps=True), self.cfg) for i in range(3)]
        self.split = split_records(self.records, self.cfg, "staged_fixture")
        self.stats = self.fixture.stats(self.split)
        self.contract = cache_contract(self.cfg, self.split, self.stats, 90)
        self.source, self.root = self.fixture.root / "light", self.fixture.root / "staged"
        self.names = self.split["splits"]["train"] + self.split["splits"]["val"]
        for record in self.records:
            prepare_light(record, self.cfg, self.source, self.split["split_hash"])
            if record['episode'] in self.names:
                cache_episode(record["episode"], self.source, self.root, self.cfg, self.split, self.contract,
                              FakeEncoder(), "cpu", batch_size=2, save_jpeg=False)
        self.ctx = {"contract": self.contract, "data_config": self.cfg, "stats": self.stats,
                    "splits": self.split, "storage_mode": "features_only_v1"}
        self.ready = {"contract_hash": digest(self.contract), "episodes": self.names, "contains_test": False}
        write_json(self.root / "READY_FROZEN.json", self.ready)
        self.checkpoint = self.fixture.root / "frozen.pt"
        self.saved = {"epoch": 0, "smoke": False, "model": {"dummy": torch.zeros(1)},
                      "metadata": {"stage": "frozen", "model_config": {"backbone": "mcr_resnet50", "lr_backbone": 0.},
                                   "feature_contract": self.contract, "data_config": self.cfg,
                                   "normalization": self.stats}}
        torch.save(self.saved, self.checkpoint)
        self.origin = validate_frozen_checkpoint(self.checkpoint, self.ctx)

    def materialize(self, name, **kwargs):
        return materialize_jpeg_episode(name, self.source, self.root, self.cfg, self.split,
            self.contract, "mp4_episode_cache_v1", batch_size=2, **kwargs)

    def test_feature_only_equals_combined_and_never_saves_jpeg(self):
        name = self.names[0]
        folder = self.root / name
        meta = check_episode(folder, self.cfg, self.split, self.contract, full=True, stage="frozen")
        self.assertEqual(meta["image_storage"], FEATURE_ONLY)
        self.assertFalse((folder / "rgb.sqlite").exists())
        combined = self.fixture.root / "combined"
        cache_episode(name, self.source, combined, self.cfg, self.split, self.contract, FakeEncoder(), "cpu", batch_size=2)
        for filename in ("features.npy", "scales.npy", "state.npy", "action.npy"):
            np.testing.assert_array_equal(np.load(folder / filename), np.load(combined / name / filename))
        require_ready(self.root, self.ctx, "frozen")
        with self.assertRaises(ValueError):require_ready(self.root, self.ctx, "full")
        with self.assertRaises(ValueError):
            FeatureACTDataset(self.root, self.split, self.stats, self.cfg, "train", feature_contract=self.contract, stage="full")
        self.assertEqual(cache_episode(name, self.source, self.root, self.cfg, self.split, self.contract,
            FakeEncoder(), "cpu", save_jpeg=False, resume=True)[1], "SKIP_VERIFIED")

    def test_jpeg_conversion_release_preserves_raw_numeric_and_full_loader(self):
        frozen = FeatureACTDataset(self.root, self.split, self.stats, self.cfg, "train", feature_contract=self.contract)
        expected = frozen[0]
        del frozen
        raw_hashes = {p: file_hash(p) for record in self.records for p in Path(record["path"]).rglob('*') if p.is_file()}
        numeric_hashes = {self.root/name/f: file_hash(self.root/name/f) for name in self.names
                          for f in ("state.npy", "action.npy", "timestamps_ms.npy", "segment_ids.npy")}
        for name in self.names:self.materialize(name)
        with self.assertRaises(ValueError):
            release_feature_episode(self.names[0], self.root, self.cfg, self.split, self.contract, self.origin)
        (self.root / "READY_FROZEN.json").unlink()
        for name in self.names:
            freed = release_feature_episode(name, self.root, self.cfg, self.split, self.contract, self.origin)
            self.assertGreater(freed, 0)
            self.assertFalse((self.root/name/'features.npy').exists())
            self.assertFalse((self.root/name/'scales.npy').exists())
            self.assertEqual(read_json(self.root/name/'episode.json')['image_storage'], JPEG_ONLY)
            check_episode(self.root/name, self.cfg, self.split, self.contract, stage="full", full=True)
            self.assertEqual(release_feature_episode(name, self.root, self.cfg, self.split, self.contract, self.origin), 0)
        write_json(self.root / "READY_FULL.json", self.ready)
        require_ready(self.root, self.ctx, "full")
        with self.assertRaises(ValueError):require_ready(self.root, self.ctx, "frozen")
        full = FeatureACTDataset(self.root, self.split, self.stats, self.cfg, "train", feature_contract=self.contract, stage="full")
        sample = full[0]
        self.assertEqual(sample['images'].shape, (8,3,224,224))
        np.testing.assert_array_equal(sample['state'],expected['state'])
        np.testing.assert_array_equal(sample['actions'],expected['actions'])
        np.testing.assert_array_equal(sample['is_pad'],expected['is_pad'])
        # Exercise real ResNet/ACT gradients after the feature files are absent.
        from src.models.mcr_act import MCRACTPolicy, PREPROCESS
        from src.training.pretrain import run_epoch
        from torch.utils.data import DataLoader
        model={**self.fixture.model,'backbone':'mcr_resnet50','feature_preprocess':PREPROCESS,
               'hidden_dim':32,'dim_feedforward':64,'enc_layers':1,'dec_layers':1,
               'nheads':4,'dropout':0.,'lr_backbone':1e-6}
        torch.set_num_threads(2)
        policy=MCRACTPolicy(model,self.cfg)
        metrics=run_epoch(policy,DataLoader(full,batch_size=1),'cpu',optimizer=policy.configure_optimizer(),max_batches=1)
        self.assertEqual(metrics['samples'],1)
        self.assertIsNotNone(policy.model.backbones[0][0].body.conv1.weight.grad)
        for path,sha in {**raw_hashes,**numeric_hashes}.items():self.assertEqual(file_hash(path),sha)
        self.assertEqual(jpeg_storage_projection(self.root,self.ctx)['feature_releasable_bytes'],0)

    def test_bad_jpeg_never_deletes_feature_files(self):
        name = self.names[0];self.materialize(name)
        folder = self.root/name
        (self.root/'READY_FROZEN.json').unlink()
        before = {f:file_hash(folder/f) for f in ('features.npy','scales.npy')}
        with closing(sqlite3.connect(folder/'rgb.sqlite')) as db:
            db.execute('DELETE FROM rgb WHERE tick=0 AND camera=0');db.commit()
        with self.assertRaises(ValueError):
            release_feature_episode(name,self.root,self.cfg,self.split,self.contract,self.origin)
        for filename,sha in before.items():self.assertEqual(file_hash(folder/filename),sha)

    def test_missing_raw_and_jpeg_budget_preserve_frozen_features(self):
        name=self.names[0];folder=self.root/name
        with self.assertRaises(OSError):self.materialize(name,max_jpeg_bytes=1)
        self.assertTrue((folder/'features.npy').is_file())
        self.assertTrue((folder/'JPEG_ERROR.json').is_file())
        self.assertFalse((folder/'rgb.sqlite').exists())
        with self.assertRaises(FileExistsError):self.materialize(name)
        other=self.names[1]
        raw=Path(next(r for r in self.records if r['episode']==other)['path'])
        (raw/f"cam_{self.cfg['camera_serials'][0]}/color.mp4").unlink()
        with self.assertRaises((OSError,ValueError)):self.materialize(other)
        self.assertTrue((self.root/other/'features.npy').is_file())

    def test_wrong_checkpoint_rejected_and_interrupted_release_resumes(self):
        torch.save({**self.saved,'smoke':True},self.checkpoint)
        with self.assertRaises(ValueError):validate_frozen_checkpoint(self.checkpoint,self.ctx)
        torch.save(self.saved,self.checkpoint)
        wrong={**self.ctx,'stats':{**self.stats,'modified':True}}
        with self.assertRaises(ValueError):validate_frozen_checkpoint(self.checkpoint,wrong)
        name=self.names[0];self.materialize(name)
        (self.root/'READY_FROZEN.json').unlink()
        folder=self.root/name
        meta=read_json(folder/'episode.json')
        write_json(folder/'episode.json',{**meta,'image_storage':JPEG_ONLY,'feature_release_status':'in_progress'})
        (folder/'features.npy').unlink()  # Simulate a crash after the first verified unlink.
        freed=release_feature_episode(name,self.root,self.cfg,self.split,self.contract,self.origin)
        self.assertGreater(freed,0)
        self.assertFalse((folder/'scales.npy').exists())
        self.assertEqual(read_json(folder/'episode.json')['feature_release_status'],'complete')

    def test_cli_pilot_build_ready_and_resume_with_exclusive_lock_calls(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from scripts import prepare_mcr_jpeg as cli
        run={'processed_root':'light','stats':'stats.json','splits':'splits.json','cache_source_backend':'mp4_episode_cache_v1'}
        write_json(self.fixture.root/'run.json',run)
        write_json(self.fixture.root/'stats.json',self.stats)
        write_json(self.fixture.root/'splits.json',self.split)
        ctx={**self.ctx,'source_config':'run.json','source_root':str(self.source.resolve())}
        write_json(self.root/'context.json',ctx)
        context_hash=file_hash(self.root/'context.json')
        class LockAPI:
            LOCK_EX=1;LOCK_NB=4
            calls=[]
            @classmethod
            def flock(cls, handle, mode):cls.calls.append(mode)
        def execute(command,*extra):
            argv=['prepare_mcr_jpeg.py',command,'--cache-root','staged','--frozen-checkpoint','frozen.pt',
                  '--home-used-gib','0','--home-quota-tb','0.1',*extra]
            with patch.object(cli,'ROOT',self.fixture.root),patch.object(sys,'argv',argv),patch.dict(sys.modules,{'fcntl':LockAPI}),redirect_stdout(io.StringIO()):
                cli.main()
        execute('pilot')
        self.assertTrue((self.root/'READY_FROZEN.json').exists())
        self.assertTrue((self.root/self.names[0]/'features.npy').exists())
        execute('build','--release-features')
        self.assertFalse((self.root/'READY_FROZEN.json').exists())
        require_ready(self.root,ctx,'full')
        self.assertEqual(file_hash(self.root/'context.json'),context_hash)
        execute('build','--release-features')
        require_ready(self.root,ctx,'full')
        self.assertEqual(LockAPI.calls,[5,5,5])


if __name__ == '__main__':unittest.main()
