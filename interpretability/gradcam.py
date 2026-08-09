"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Produces a heatmap showing which regions of an input image most influenced
the model's prediction -- the "why did the model think this" visualization
your project requires.

Works identically for BOTH architectures used across modalities (MedicalCNN
for X-ray/MRI, TransferResNetCNN for CT) because both expose their last
convolutional block as `model.features_final` (see models/cnn_model.py).
This module never needs to know which architecture it's looking at.
"""

import cv2
import numpy as np
import torch
from PIL import Image

import config
from utils.inference import predict_class_index


class GradCAM:
    """
    Hooks into a target conv layer to capture:
      - the forward activations (what features fired)
      - the backward gradients (how much each feature mattered for the
        predicted class)
    Combining them (gradient-weighted sum of activation maps) produces the
    class activation map: a coarse heatmap over the last conv layer's
    spatial grid, highlighting regions that drove the prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: preprocessed image, shape (1, 3, H, W), already on the
            correct device.
        target_class: which class's score to explain. If None, uses the
            model's own top prediction (the common case: "why did you predict
            this?"). Pass an explicit index to instead ask "what would make
            this look like class X?" -- useful for understanding misclassifications.

        Returns: (cam, target_class) where cam is a (h, w) numpy array in
        [0, 1], h/w matching the target layer's spatial resolution (16x16 for
        MedicalCNN, 4x4 for TransferResNetCNN at IMAGE_SIZE=128).
        """
        self.model.eval()

        # Explicitly require grad on the INPUT (not just model parameters).
        # This matters specifically for TransferResNetCNN (CT), where the
        # early layers are frozen (requires_grad=False) -- without this,
        # gradients wouldn't flow back to features_final at all, since every
        # parameter upstream of it has requires_grad=False. Setting it on
        # the input guarantees the activation tensor requires grad
        # regardless of which layers are frozen.
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # activations/gradients shape: (1, C, h, w) -> drop batch dim
        activations = self.activations[0]
        gradients = self.gradients[0]

        # Global-average-pool the gradients over spatial dims -> one
        # importance weight per channel. This is the core Grad-CAM idea:
        # channels whose activations correlate strongly with the target
        # class's gradient get a high weight.
        weights = gradients.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)  # only positive influence, per the Grad-CAM paper
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy(), target_class

    def remove_hooks(self):
        """Call this when done to avoid leaking hooks if reusing the model elsewhere."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(original_pil_image: Image.Image, cam: np.ndarray, image_size: int, alpha: float = 0.45) -> Image.Image:
    """
    Resizes the (small, coarse) CAM up to the display image size and blends
    it as a color heatmap over the original image.

    alpha: heatmap opacity. 0.45 keeps the underlying scan clearly visible
    underneath the color overlay -- important since a radiologist/user needs
    to see the actual anatomy, not just the heatmap.
    """
    heatmap = cv2.resize(cam, (image_size, image_size))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR, red=high importance
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original = original_pil_image.convert("RGB").resize((image_size, image_size))
    original_np = np.array(original).astype(np.uint8)

    overlay = (alpha * heatmap_color + (1 - alpha) * original_np).astype(np.uint8)
    return Image.fromarray(overlay)


def explain_prediction(model: torch.nn.Module, pil_image: Image.Image, transform, device, decision_threshold: float = 0.5):
    """
    High-level convenience function the dashboard (Step 6) calls directly.

    pil_image: the ORIGINAL uploaded image (before any resizing/normalizing) --
        this function handles preprocessing internally so the overlay is
        built on the true original image, not a lower-quality intermediate.
    transform: the eval-mode transform from utils.preprocessing.get_transforms
        (must be train=False -- no augmentation for inference).
    decision_threshold: for binary modalities, the calibrated threshold from
        the checkpoint (see training/train.py's threshold calibration step).
        Ignored for multiclass models (predict_class_index falls back to
        argmax automatically when there are more than 2 classes).

    IMPORTANT: we compute the final predicted class FIRST (respecting the
    threshold), then generate Grad-CAM for exactly that class. This ensures
    the heatmap always explains the prediction actually shown to the user --
    if we let Grad-CAM default to raw argmax while displaying a
    threshold-adjusted class elsewhere, the two could disagree and the
    "explanation" would be for the wrong class.

    Returns a dict with everything the dashboard needs to display:
        overlay_image: PIL Image, the Grad-CAM heatmap blended onto the scan
        predicted_class_index: int
        confidence: float, softmax probability of the predicted class
        all_probabilities: numpy array, softmax probs for every class
    """
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # First pass (no grad): get probabilities to decide the final class.
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()

    predicted_class_index = predict_class_index(probabilities, decision_threshold)

    # Second pass (with grad, inside GradCAM): explain THAT specific class.
    cam_engine = GradCAM(model, model.features_final)
    cam, _ = cam_engine.generate(input_tensor, target_class=predicted_class_index)
    cam_engine.remove_hooks()

    confidence = float(probabilities[predicted_class_index])
    overlay_image = overlay_heatmap(pil_image, cam, image_size=config.IMAGE_SIZE)

    return {
        "overlay_image": overlay_image,
        "predicted_class_index": predicted_class_index,
        "confidence": confidence,
        "all_probabilities": probabilities,
    }


if __name__ == "__main__":
    # Local smoke test: run this directly in PyCharm once at least one
    # modality has a trained checkpoint. Picks the first modality with an
    # existing checkpoint file, runs Grad-CAM on one sample test image, and
    # saves the overlay so you can visually confirm it looks reasonable.
    import os
    from models.cnn_model import build_model
    from training.dataset import build_datasets
    from utils.preprocessing import get_transforms

    tested_any = False

    for modality, cfg in config.MODALITIES.items():
        if not os.path.exists(cfg["checkpoint"]):
            print(f"[{modality}] No checkpoint yet at {cfg['checkpoint']} -- skipping. "
                  f"Run training/train.py for this modality first.")
            continue

        tested_any = True
        print(f"[{modality}] Loading checkpoint and testing Grad-CAM...")

        checkpoint = torch.load(cfg["checkpoint"], map_location=config.DEVICE)
        use_pretrained = bool(cfg.get("use_pretrained", False))
        model = build_model(num_classes=len(cfg["classes"]), device=config.DEVICE, pretrained=use_pretrained)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, _, test_ds = build_datasets(modality)
        sample_path, true_label = test_ds.samples[0]
        pil_image = Image.open(sample_path).convert("RGB")

        eval_transform = get_transforms(config.IMAGE_SIZE, train=False)
        result = explain_prediction(model, pil_image, eval_transform, config.DEVICE)

        predicted_class = cfg["classes"][result["predicted_class_index"]]
        true_class = cfg["classes"][true_label]
        print(f"  True class: {true_class} | Predicted: {predicted_class} | "
              f"Confidence: {result['confidence']:.4f}")

        out_path = f"gradcam_test_{modality}.png"
        result["overlay_image"].save(out_path)
        print(f"  Saved overlay to {out_path}")

    if not tested_any:
        print("\nNo checkpoints found for any modality yet. Train at least one "
              "modality (training/train.py) before testing Grad-CAM.")