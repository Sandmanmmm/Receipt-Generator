"""
LayoutLMv3 Multi-Head Model
Token-level NER + Table Detection + Cell Attribute Classification
Production-ready architecture for invoice/PO extraction
"""
import torch
import torch.nn as nn
from transformers import LayoutLMv3Model, LayoutLMv3PreTrainedModel
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MultiHeadOutput:
    """Output from multi-head model"""
    loss: Optional[torch.Tensor] = None
    ner_loss: Optional[torch.Tensor] = None
    table_loss: Optional[torch.Tensor] = None
    attr_loss: Optional[torch.Tensor] = None
    ner_logits: Optional[torch.Tensor] = None
    table_logits: Optional[torch.Tensor] = None
    attr_logits: Optional[torch.Tensor] = None
    crf_emissions: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class LayoutLMv3MultiHead(LayoutLMv3PreTrainedModel):
    """
    LayoutLMv3 with multiple classification heads:
    1. NER head (token classification with optional CRF)
    2. Table detection head (B-TABLE/I-TABLE/O)
    3. Cell attribute head (column type classification)
    """
    
    def __init__(self, config, num_ner_labels: int = 73, use_crf: bool = True):
        """
        Initialize multi-head model
        
        Args:
            config: LayoutLMv3Config
            num_ner_labels: Number of NER labels (including O)
            use_crf: Whether to use CRF layer for NER
        """
        super().__init__(config)
        self.num_ner_labels = num_ner_labels
        self.use_crf = use_crf
        
        # Backbone
        self.layoutlmv3 = LayoutLMv3Model(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # NER head
        self.ner_classifier = nn.Linear(config.hidden_size, num_ner_labels)
        
        # CRF layer (optional)
        if use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(num_ner_labels, batch_first=True)
            except ImportError:
                raise ImportError(
                    "torchcrf not installed. Install with: pip install pytorch-crf"
                )
        else:
            self.crf = None
        
        # Table detection head (3 labels: O, B-TABLE, I-TABLE)
        self.table_classifier = nn.Linear(config.hidden_size, 3)
        
        # Cell attribute head (column type classification)
        # Predicts probabilities for: quantity, price, description
        self.cell_attr_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 3)  # [qty_prob, price_prob, desc_prob]
        )
        
        # Loss weights (from config)
        self.ner_loss_weight = 1.0
        self.table_loss_weight = 0.7
        self.attr_loss_weight = 0.5
        
        # Initialize weights
        self.post_init()
    
    def set_loss_weights(self, ner_weight: float = 1.0, 
                        table_weight: float = 0.7, 
                        attr_weight: float = 0.5):
        """Set multi-task loss weights"""
        self.ner_loss_weight = ner_weight
        self.table_loss_weight = table_weight
        self.attr_loss_weight = attr_weight
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        table_labels: Optional[torch.Tensor] = None,
        attr_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MultiHeadOutput:
        """
        Forward pass with multi-task outputs
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            bbox: Bounding boxes [batch, seq_len, 4]
            pixel_values: Image features [batch, channels, height, width]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            labels: NER labels [batch, seq_len]
            table_labels: Table detection labels [batch, seq_len]
            attr_labels: Cell attribute labels [batch, seq_len, 3]
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict
        
        Returns:
            MultiHeadOutput with losses and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through backbone
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # NER head
        ner_logits = self.ner_classifier(sequence_output)  # [batch, seq_len, num_ner_labels]
        
        # Table detection head
        table_logits = self.table_classifier(sequence_output)  # [batch, seq_len, 3]
        
        # Cell attribute head
        attr_logits = self.cell_attr_head(sequence_output)  # [batch, seq_len, 3]
        
        # Calculate losses
        total_loss = None
        ner_loss = None
        table_loss = None
        attr_loss = None
        
        if labels is not None:
            # NER loss
            if self.use_crf and self.crf is not None:
                # CRF loss (negative log-likelihood)
                # Mask padding tokens
                mask = attention_mask.bool()
                ner_loss = -self.crf(ner_logits, labels, mask=mask, reduction='mean')
            else:
                # Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ner_loss = loss_fct(
                    ner_logits.view(-1, self.num_ner_labels),
                    labels.view(-1)
                )
            
            total_loss = self.ner_loss_weight * ner_loss
        
        if table_labels is not None:
            # Table detection loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            table_loss = loss_fct(
                table_logits.view(-1, 3),
                table_labels.view(-1)
            )
            
            if total_loss is not None:
                total_loss += self.table_loss_weight * table_loss
            else:
                total_loss = self.table_loss_weight * table_loss
        
        if attr_labels is not None:
            # Cell attribute loss (multi-label BCE)
            loss_fct = nn.BCEWithLogitsLoss()
            attr_loss = loss_fct(attr_logits, attr_labels.float())
            
            if total_loss is not None:
                total_loss += self.attr_loss_weight * attr_loss
            else:
                total_loss = self.attr_loss_weight * attr_loss
        
        return MultiHeadOutput(
            loss=total_loss,
            ner_loss=ner_loss,
            table_loss=table_loss,
            attr_loss=attr_loss,
            ner_logits=ner_logits,
            table_logits=table_logits,
            attr_logits=attr_logits,
            crf_emissions=ner_logits if self.use_crf else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode with CRF decoding
        
        Args:
            input_ids: Token IDs
            bbox: Bounding boxes
            pixel_values: Image features
            attention_mask: Attention mask
        
        Returns:
            Dictionary with predictions
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                bbox=bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            
            # NER predictions
            if self.use_crf and self.crf is not None:
                # CRF Viterbi decoding
                mask = attention_mask.bool() if attention_mask is not None else None
                ner_predictions = self.crf.decode(outputs.ner_logits, mask=mask)
                ner_predictions = torch.tensor(ner_predictions).to(input_ids.device)
            else:
                # Argmax
                ner_predictions = outputs.ner_logits.argmax(dim=-1)
            
            # Table predictions
            table_predictions = outputs.table_logits.argmax(dim=-1)
            
            # Cell attribute predictions (sigmoid for multi-label)
            attr_predictions = torch.sigmoid(outputs.attr_logits)
            
            return {
                'ner_predictions': ner_predictions,
                'ner_logits': outputs.ner_logits,
                'table_predictions': table_predictions,
                'table_logits': outputs.table_logits,
                'attr_predictions': attr_predictions,
                'attr_logits': outputs.attr_logits,
            }
    
    def get_ner_predictions_with_confidence(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get NER predictions with confidence scores
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        predictions = self.predict(input_ids, bbox, pixel_values, attention_mask)
        
        # Get confidence scores
        ner_probs = torch.softmax(predictions['ner_logits'], dim=-1)
        confidence_scores = ner_probs.max(dim=-1).values
        
        return predictions['ner_predictions'], confidence_scores


def create_model(
    pretrained_name: str = "microsoft/layoutlmv3-base",
    num_ner_labels: int = 73,
    use_crf: bool = True,
    device: str = "cuda"
) -> LayoutLMv3MultiHead:
    """
    Create and initialize multi-head model
    
    Args:
        pretrained_name: HuggingFace model name
        num_ner_labels: Number of NER labels
        use_crf: Whether to use CRF layer
        device: Device to load model on
    
    Returns:
        LayoutLMv3MultiHead model
    """
    from transformers import LayoutLMv3Config
    
    config = LayoutLMv3Config.from_pretrained(pretrained_name)
    model = LayoutLMv3MultiHead(config, num_ner_labels=num_ner_labels, use_crf=use_crf)
    
    # Load pretrained weights (backbone only)
    model.layoutlmv3 = LayoutLMv3Model.from_pretrained(pretrained_name)
    
    model = model.to(device)
    
    return model


if __name__ == '__main__':
    # Test model instantiation
    print("Testing LayoutLMv3MultiHead model...")
    
    model = create_model(
        pretrained_name="microsoft/layoutlmv3-base",
        num_ner_labels=73,
        use_crf=True,
        device="cpu"
    )
    
    print(f"✓ Model created successfully")
    print(f"  NER labels: {model.num_ner_labels}")
    print(f"  Using CRF: {model.use_crf}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    bbox = torch.randint(0, 1000, (batch_size, seq_len, 4))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 73, (batch_size, seq_len))
    
    output = model(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"\n✓ Forward pass successful")
    print(f"  NER logits shape: {output.ner_logits.shape}")
    print(f"  Table logits shape: {output.table_logits.shape}")
    print(f"  Attr logits shape: {output.attr_logits.shape}")
    print(f"  Total loss: {output.loss.item():.4f}")
    print(f"  NER loss: {output.ner_loss.item():.4f}")
