from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as Functional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss.PedalBoundaryFocalLoss import PedalBoundaryFocalLoss


def _config(gamma=2.0):
    return types.SimpleNamespace(
        training=types.SimpleNamespace(
            pedal_boundary_focal_gamma=gamma,
        )
    )


def test_pedal_boundary_focal_loss_accepts_soft_targets_and_mask():
    loss_fn = PedalBoundaryFocalLoss(_config())
    outputs = torch.tensor([[0.0, 2.0, -2.0, 0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.946, 0.25, 0.0]])
    mask = torch.tensor([[1, 1, 0, 1]])

    loss = loss_fn(outputs, targets, mask)

    assert torch.isfinite(loss)
    loss.backward()
    assert outputs.grad is not None
    assert outputs.grad[0, 2].item() == 0.0


def test_pedal_boundary_focal_loss_accepts_unmasked_inputs():
    loss_fn = PedalBoundaryFocalLoss(_config(gamma=1.5))
    outputs = torch.tensor([[0.0, 1.0]])
    targets = torch.tensor([[1.0, 0.5]])

    loss = loss_fn(outputs, targets, None)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_quality_focal_loss_focuses_soft_targets_by_prediction_error():
    loss_fn = PedalBoundaryFocalLoss(_config(gamma=2.0))
    target = torch.tensor([[0.5]])
    close_output = torch.logit(torch.tensor([[0.5]]))
    low_output = torch.logit(torch.tensor([[0.1]]))
    high_output = torch.logit(torch.tensor([[0.9]]))

    close_loss = loss_fn(close_output, target, None)
    low_loss = loss_fn(low_output, target, None)
    high_loss = loss_fn(high_output, target, None)

    assert close_loss.item() < low_loss.item()
    assert close_loss.item() < high_loss.item()
    assert close_loss.item() == 0.0


def test_quality_focal_loss_matches_no_alpha_binary_focal_cases():
    loss_fn = PedalBoundaryFocalLoss(_config(gamma=2.0))
    probabilities = torch.tensor([[0.2, 0.8]])
    outputs = torch.logit(probabilities)
    targets = torch.tensor([[0.0, 1.0]])

    loss = loss_fn(outputs, targets, None)
    bce = Functional.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    expected = torch.mean(torch.tensor([[0.2 ** 2, (1.0 - 0.8) ** 2]]) * bce)

    torch.testing.assert_close(loss, expected)


def test_focal_experiment_config_keeps_state_head_on_bce():
    project_root = Path(__file__).resolve().parents[1]
    base_config = (project_root / "config/base_config.yaml").read_text()
    focal_config = (
        project_root / "config/experiment_T5_V4_HierarchyPool_GaussianBoundary_Focal.yaml"
    ).read_text()

    assert "loss_pedal_state:  [pedal_frame_logits,  pedal_frame_target,  loss.PedalFrameBCELoss" in base_config
    assert "loss_pedal_onset:  [pedal_onset_logits,  pedal_onset_target,  loss.PedalBoundaryFocalLoss" in focal_config
    assert "loss_pedal_offset: [pedal_offset_logits, pedal_offset_target, loss.PedalBoundaryFocalLoss" in focal_config
    assert "pedal_boundary_focal_alpha" not in base_config
    assert "pedal_boundary_focal_alpha" not in focal_config
    assert "loss_pedal_state" not in focal_config
