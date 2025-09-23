"""Test script for LRCN model."""

import torch
from src.medvqa.models import LRCN


def test_lrcn_model():
    """Test LRCN model with dummy data."""
    print("Testing LRCN Model...")

    # Create model
    model = LRCN(
        hidden_dim=512,
        num_attention_layers=3,  # Smaller for testing
        num_classes=100,  # Dummy vocabulary size
        use_lrm=True,
    )

    print(f"Model created on device: {model.device}")

    # Create dummy data
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    questions = [
        "What is shown in this medical image?",
        "Is there any abnormality visible?",
    ]

    print(f"Input images shape: {images.shape}")
    print(f"Questions: {questions}")

    # Test forward pass
    try:
        outputs = model(images, questions, return_attention=True)

        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Number of attention layers: {len(outputs['attention_weights'])}")

        # Test prediction
        predictions = model.predict(images, questions)
        print(f"Predictions shape: {predictions['predictions'].shape}")
        print(f"Confidence scores: {predictions['confidence']}")

        # Test attention maps
        attention_maps = model.get_attention_maps(images, questions)
        print(
            f"Visual self-attention shape: {attention_maps['visual_self_attention'].shape}"
        )
        print(
            f"Visual guided-attention shape: {attention_maps['visual_guided_attention'].shape}"
        )

        # Count parameters
        param_count = model.count_parameters()
        print(f"Total parameters: {param_count['total']:,}")
        print(f"Trainable parameters: {param_count['trainable']:,}")

        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_lrcn_model()
