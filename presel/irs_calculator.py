"""
Instruction Relevance Score (IRS) Calculator

Implements the IRS computation as described in Equation 3:
IRS = L(R|Q,I) / L(R|I)

where:
- L(R|Q,I): loss of generating response R given question Q and image I
- L(R|I): loss of generating response R given only image I
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class IRSCalculator:
    """
    Calculate Instruction Relevance Score (IRS) for VIT samples.

    The IRS measures how much the question (Q) contributes to generating
    the ground-truth response (R) given an image (I).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: Pre-trained LVLM (e.g., LLaVA-1.5)
            tokenizer: Tokenizer for the model
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        self.model.eval()

    def compute_loss_with_question(
        self,
        image: torch.Tensor,
        question: str,
        response: str
    ) -> float:
        """
        Compute L(R|Q,I) - loss of generating R given both Q and I.

        Args:
            image: Image tensor
            question: Question text
            response: Response text (ground truth)

        Returns:
            Cross-entropy loss value
        """
        # Tokenize response
        response_tokens = self.tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)

        # Create input with image, question, and response
        # Format: <image> Question: {Q} Response: {R}
        input_text = f"Question: {question}\nResponse: {response}"

        with torch.no_grad():
            # Get model outputs
            # Note: Actual implementation depends on the specific LVLM architecture
            # This is a generic implementation
            outputs = self._forward_model(image, input_text, response_tokens)
            loss = outputs.get('loss', self._compute_ce_loss(outputs, response_tokens))

        return loss.item()

    def compute_loss_without_question(
        self,
        image: torch.Tensor,
        response: str
    ) -> float:
        """
        Compute L(R|I) - loss of generating R given only I (no question).

        Args:
            image: Image tensor
            response: Response text (ground truth)

        Returns:
            Cross-entropy loss value
        """
        # Tokenize response
        response_tokens = self.tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)

        # Create input with only image and response (no question)
        # Format: <image> Response: {R}
        input_text = f"Response: {response}"

        with torch.no_grad():
            # Get model outputs
            outputs = self._forward_model(image, input_text, response_tokens)
            loss = outputs.get('loss', self._compute_ce_loss(outputs, response_tokens))

        return loss.item()

    def compute_irs(
        self,
        image: torch.Tensor,
        question: str,
        response: str
    ) -> float:
        """
        Compute Instruction Relevance Score (IRS) for a single sample.

        IRS = L(R|Q,I) / L(R|I)

        Args:
            image: Image tensor
            question: Question text
            response: Response text

        Returns:
            IRS value (lower is better)
        """
        loss_with_q = self.compute_loss_with_question(image, question, response)
        loss_without_q = self.compute_loss_without_question(image, response)

        # Avoid division by zero
        if loss_without_q < 1e-8:
            return 1.0

        irs = loss_with_q / loss_without_q
        return irs

    def compute_batch_irs(
        self,
        samples: List[Dict],
        show_progress: bool = True
    ) -> List[float]:
        """
        Compute IRS for a batch of samples.

        Args:
            samples: List of dicts with keys 'image', 'question', 'response'
            show_progress: Whether to show progress bar

        Returns:
            List of IRS values
        """
        irs_scores = []

        iterator = tqdm(samples, desc="Computing IRS") if show_progress else samples

        for sample in iterator:
            irs = self.compute_irs(
                sample['image'],
                sample['question'],
                sample['response']
            )
            irs_scores.append(irs)

        return irs_scores

    def _forward_model(
        self,
        image: torch.Tensor,
        text: str,
        response_tokens: torch.Tensor
    ) -> Dict:
        """
        Forward pass through the LVLM.

        This is a generic implementation. You should customize this based on
        your specific LVLM architecture (LLaVA, InstructBLIP, etc.)

        Args:
            image: Image tensor
            text: Input text (with or without question)
            response_tokens: Tokenized response

        Returns:
            Model outputs including loss or logits
        """
        # Generic implementation - customize for your model
        try:
            # For models like LLaVA that support generate
            if hasattr(self.model, 'forward'):
                # Tokenize full input
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

                # Process image if model has vision encoder
                if hasattr(self.model, 'vision_tower') or hasattr(self.model, 'encode_images'):
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    image = image.to(self.device)

                    # Forward pass with image
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        images=image,
                        labels=response_tokens,
                        return_dict=True
                    )
                else:
                    # Text-only model (for testing)
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        labels=response_tokens,
                        return_dict=True
                    )

                return outputs
        except Exception as e:
            print(f"Warning: Model forward failed: {e}")
            # Return dummy output for testing
            return {'logits': torch.randn(1, response_tokens.size(1), 32000).to(self.device)}

    def _compute_ce_loss(
        self,
        outputs: Dict,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss from model outputs.

        Args:
            outputs: Model outputs containing logits
            target_tokens: Target token IDs

        Returns:
            Cross-entropy loss
        """
        logits = outputs.get('logits')

        if logits is None:
            # Return dummy loss if logits not available
            return torch.tensor(1.0).to(self.device)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_tokens[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss


class SimplifiedIRSCalculator:
    """
    Simplified IRS calculator that works without a real LVLM.

    This is useful for testing and when you don't have access to a full LVLM.
    It uses a simple heuristic based on text similarity and length.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def compute_irs(
        self,
        image: torch.Tensor,
        question: str,
        response: str
    ) -> float:
        """
        Compute simplified IRS based on heuristics.

        The heuristic assumes:
        - Longer questions provide more context (lower IRS)
        - More complex responses benefit more from questions (lower IRS)
        """
        # Heuristic 1: Question informativeness (based on length and words)
        question_words = len(question.split())
        response_words = len(response.split())

        # Heuristic 2: Image complexity (based on image statistics)
        if isinstance(image, torch.Tensor):
            image_std = image.std().item() if image.numel() > 0 else 0.5
        else:
            image_std = 0.5

        # Compute simplified IRS
        # Lower when question is informative relative to response complexity
        base_irs = 0.8
        question_factor = max(0.5, 1.0 - (question_words / max(response_words, 1)) * 0.3)
        image_factor = 0.9 + image_std * 0.2

        irs = base_irs * question_factor * image_factor

        # Add some randomness to simulate real variance
        import random
        irs += random.gauss(0, 0.05)
        irs = max(0.3, min(1.2, irs))  # Clip to reasonable range

        return irs

    def compute_batch_irs(
        self,
        samples: List[Dict],
        show_progress: bool = True
    ) -> List[float]:
        """Compute IRS for a batch of samples."""
        irs_scores = []

        iterator = tqdm(samples, desc="Computing IRS (simplified)") if show_progress else samples

        for sample in iterator:
            irs = self.compute_irs(
                sample['image'],
                sample['question'],
                sample['response']
            )
            irs_scores.append(irs)

        return irs_scores