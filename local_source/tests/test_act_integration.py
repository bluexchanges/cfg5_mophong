import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from src.data.common import ROOT, read_json, load_config
    from src.models.rh20t_act import RH20TACTPolicy, masked_l1, verify_upstream


@unittest.skipUnless(TORCH, "Cần requirements-act.txt")
class ACTIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = read_json(ROOT / "configs/act/rh20t_base8.json")
        cls.cfg.update(hidden_dim=32, dim_feedforward=64, enc_layers=1, dec_layers=1, nheads=4, dropout=0.)
        cls.data = load_config(ROOT / cls.cfg["data_config"])
        cls.data.update(image_height=32, image_width=32)
        torch.manual_seed(12)
        cls.policy = RH20TACTPolicy(cls.cfg, cls.data).eval()
        cls.state = torch.zeros(2,8)
        cls.images = torch.randn(2,8,3,32,32)
        cls.actions = torch.randn(2,20,8)
        cls.mask = torch.ones(2,20,dtype=torch.bool)
        cls.mask[0,:3] = False; cls.mask[1,:7] = False

    def test_upstream_hashes_unchanged(self):
        verify_upstream()

    def test_loss_denominator_and_empty_sample(self):
        target = torch.zeros(2,20,8); pred = torch.ones_like(target)
        self.assertEqual(float(masked_l1(pred,target,self.mask)), 1.)
        pred[self.mask] = 9000
        self.assertEqual(float(masked_l1(pred,target,self.mask)), 1.)
        with self.assertRaises(ValueError): masked_l1(pred,target,torch.ones_like(self.mask))

    def test_padding_does_not_change_posterior_or_loss(self):
        poisoned = self.actions.clone(); poisoned[self.mask] = float('nan')
        with torch.no_grad():
            torch.manual_seed(1); a = self.policy(self.state,self.images,self.actions,self.mask)
            torch.manual_seed(1); b = self.policy(self.state,self.images,poisoned,self.mask)
        torch.testing.assert_close(a['prediction'],b['prediction'],rtol=0,atol=0)
        torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)
        self.assertEqual(int(a['valid_action_elements']),80)

    def test_batch_inference_module_hook_and_checkpoint_roundtrip(self):
        calls=[]
        handle=self.policy.register_forward_hook(lambda *args:calls.append(True))
        with torch.no_grad():
            prediction=self.policy(self.state,self.images)
        handle.remove()
        self.assertEqual(len(calls),1)
        self.assertEqual(tuple(prediction.shape),(2,20,8))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'untrained_state_dict.pt'
            torch.save(self.policy.state_dict(),path)
            restored=RH20TACTPolicy(self.cfg,self.data).eval()
            restored.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
            with torch.no_grad(): actual=restored(self.state,self.images)
        torch.testing.assert_close(prediction,actual,rtol=0,atol=0)

    def test_camera_identity_and_finite_backward(self):
        model=self.policy
        pos=torch.zeros(1,32,1,8)
        encoded=model.model.transformer.camera_positions(pos)
        self.assertFalse(torch.equal(encoded[:,:,:,0],encoded[:,:,:,1]))
        model.zero_grad(set_to_none=True)
        losses=model(self.state,self.images,self.actions,self.mask)
        losses['loss'].backward()
        self.assertGreater(float(model.model.transformer.camera_embedding.weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_contract_rejects_wrong_views_mask_and_chunk(self):
        with self.assertRaises(ValueError): self.policy(self.state,self.images[:,:7])
        with self.assertRaises(ValueError): self.policy(self.state,self.images,self.actions,self.mask.float())
        cfg=deepcopy(self.cfg);cfg['chunk_size']=10
        with self.assertRaises(ValueError): RH20TACTPolicy(cfg,self.data)
        cfg=deepcopy(self.cfg);cfg['nheads']=0
        with self.assertRaises(ValueError): RH20TACTPolicy(cfg,self.data)


if __name__=='__main__': unittest.main()
