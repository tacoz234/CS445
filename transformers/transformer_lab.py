"""
LLM Tokenization and Attention Visualization Lab
=================================================
A hands-on lab for exploring how Language Models tokenize text and compute attention.

Author: Claude Sonnet 4.5 and Nathan Sprague
Date: October 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TransformerLab:
    """A class for exploring tokenization and attention in small LLMs."""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the lab with a pre-trained model.
        
        Args:
            model_name: HuggingFace model name (default: "gpt2" - 124M parameters)
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            attn_implementation="eager"  # Use eager for attention output support
        )
        self.model.config.output_attentions = True  # Enable attention output
        self.model.eval()  # Set to evaluation mode
        print("done.")
    
    def tokenize(self, text: str) -> dict:
        """
        Tokenize text without displaying results.
        
        Args:
            text: Input string to tokenize
            
        Returns:
            Dictionary with tokens, token_ids, and encoding
        """
        # Tokenize
        encoding = self.tokenizer(text, return_tensors="pt")
        token_ids = encoding['input_ids'][0].tolist()
        
        # Decode each token individually
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        return {
            'text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'encoding': encoding
        }
    
    def display_tokens(self, text: str) -> None:
        """
        Tokenize text and display the results in a readable format.
        
        Args:
            text: Input string to tokenize
        """
        print(f"{'-'*70}")
        print(f"Input text: \"{text}\"")
        print(f"{'-'*70}")
        
        # Get tokenization
        token_info = self.tokenize(text)
        tokens = token_info['tokens']
        token_ids = token_info['token_ids']
        
        # Display results
        print(f"Number of tokens: {len(tokens)}")
        print(f"\nToken breakdown:")
        for i, (token, tid) in enumerate(zip(tokens, token_ids)):
            # Show special characters clearly
            display_token = repr(token) if token.strip() != token else token
            print(f"  [{i:2d}] {display_token:20s} (ID: {tid:5d})")
    
    def compute_attention(self, text: str) -> Tuple[torch.Tensor, dict]:
        """
        Compute attention patterns for the input text.
        
        Args:
            text: Input string
            
        Returns:
            Tuple of (attention_patterns, tokenization_info)
            attention_patterns shape: [num_layers, num_heads, seq_len, seq_len]
        """
        # Tokenize
        token_info = self.tokenize(text)
        encoding = token_info['encoding']
        
        # Forward pass with attention weights
        with torch.no_grad():
            outputs = self.model(**encoding, output_attentions=True)
        
        # Extract attention activations
        # attentions is a tuple of length num_layers
        # Each element has shape [batch_size, num_heads, seq_len, seq_len]
        attentions = outputs.attentions
        
        # Stack into single tensor and remove batch dimension
        attention_tensor = torch.stack(attentions).squeeze(1)
        
        return attention_tensor, token_info
    
    def visualize_attention(
        self,
        attention_pattern: torch.Tensor,
        tokens: List[str],
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Visualize a single attention pattern as a heatmap.
        
        Args:
            attention_pattern: Attention tensor [seq_len, seq_len] for a single head
            tokens: List of token strings
            figsize: Figure size for the plot
        """
        # Convert to numpy
        attn = attention_pattern.cpu().detach().numpy()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=1
        )
        
        plt.title('Attention Weights', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Key Tokens (attending to)', fontsize=12)
        plt.ylabel('Query Tokens (attending from)', fontsize=12)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_attention_heads_grid(
        self,
        attention_patterns: torch.Tensor,
        tokens: List[str],
        layer: int = 0,
        max_heads: int = 12,
        figsize: Tuple[int, int] = (20, 16)
    ):
        """
        Visualize multiple attention heads in a grid layout.
        
        Args:
            attention_patterns: Attention tensor [layers, heads, seq_len, seq_len]
            tokens: List of token strings
            layer: Which layer to visualize
            max_heads: Maximum number of heads to display
            figsize: Figure size for the plot
        """
        num_heads = min(attention_patterns.shape[1], max_heads)
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for head in range(num_heads):
            attn = attention_patterns[layer, head].cpu().numpy()
            
            ax = axes[head]
            sns.heatmap(
                attn,
                ax=ax,
                cmap='viridis',
                cbar=True,
                square=True,
                xticklabels=tokens if len(tokens) <= 10 else False,
                yticklabels=tokens if len(tokens) <= 10 else False,
                vmin=0,
                vmax=1
            )
            ax.set_title(f'Head {head}', fontsize=10, fontweight='bold')
            
            if len(tokens) <= 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Attention Patterns Across Heads - Layer {layer}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    
    def get_attention_head_weights(
        self,
        layer: int = 0,
        head: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract the Q, K, V weight matrices and biases for a specific attention head.
        
        These are LEARNED PARAMETERS - they're the same regardless of input text!
        The model learned these weights during training.
        
        Args:
            layer: Which layer to use (0-11 for GPT-2 small)
            head: Which attention head to use (0-11 for GPT-2 small)
            
        Returns:
            Tuple of (W_Q, W_K, W_V, b_Q, b_K, b_V)
            - W_Q: [d_model, d_head] - Query weight matrix for this head
            - W_K: [d_model, d_head] - Key weight matrix for this head
            - W_V: [d_model, d_head] - Value weight matrix for this head
            - b_Q: [d_head] - Query bias for this head
            - b_K: [d_head] - Key bias for this head
            - b_V: [d_head] - Value bias for this head
            
        Note:
            d_model = 768 (embedding dimension)
            d_head = 64 (dimension per head = 768 / 12)
        """
        # Get the attention layer
        attn_layer = self.model.transformer.h[layer].attn
        
        # Get dimensions
        d_model = self.model.config.n_embd  # 768
        n_head = self.model.config.n_head    # 12
        d_head = d_model // n_head           # 64
        
        # Get combined QKV weight matrix
        # GPT-2 combines Q, K, V into one matrix, so we need to split it
        # c_attn has shape [d_model, 3 * d_model] and contains Q, K, V weights
        qkv_weight = attn_layer.c_attn.weight.data  # [d_model, 3*d_model]
        qkv_bias = attn_layer.c_attn.bias.data      # [3*d_model]
        
        # Split into Q, K, V
        W_Q_all = qkv_weight[:, :d_model]           # [d_model, d_model]
        W_K_all = qkv_weight[:, d_model:2*d_model]  # [d_model, d_model]
        W_V_all = qkv_weight[:, 2*d_model:]         # [d_model, d_model]
        
        b_Q_all = qkv_bias[:d_model]                # [d_model]
        b_K_all = qkv_bias[d_model:2*d_model]       # [d_model]
        b_V_all = qkv_bias[2*d_model:]              # [d_model]
        
        # Extract weights for specific head
        # Each head gets a slice of the full weight matrix
        start_idx = head * d_head
        end_idx = (head + 1) * d_head
        
        W_Q = W_Q_all[:, start_idx:end_idx]  # [d_model, d_head] = [768, 64]
        W_K = W_K_all[:, start_idx:end_idx]  # [d_model, d_head] = [768, 64]
        W_V = W_V_all[:, start_idx:end_idx]  # [d_model, d_head] = [768, 64]
        
        b_Q = b_Q_all[start_idx:end_idx]     # [d_head] = [64]
        b_K = b_K_all[start_idx:end_idx]     # [d_head] = [64]
        b_V = b_V_all[start_idx:end_idx]     # [d_head] = [64]
        
        return W_Q, W_K, W_V, b_Q, b_K, b_V
    
    def get_input_embeddings(
        self,
        text: str,
        apply_layer_norm: bool = True
    ) -> torch.Tensor:
        """
        Get combined token + positional embeddings for input text.
        
        These are the embeddings at the INPUT to the transformer (layer 0).
        Positional embeddings are only added once at the input - not in subsequent layers!
        
        Args:
            text: Input text to get embeddings for
            apply_layer_norm: Whether to apply layer normalization (needed for attention computation!)
            
        Returns:
            embeddings: [seq_len, d_model] - Combined token + position embeddings
            
        Note:
            - Token embeddings capture word meaning
            - Position embeddings capture word position (0, 1, 2, ...)
            - Final embedding = token_emb + position_emb (element-wise addition)
            - If apply_layer_norm=True, applies layer 0's layer norm
              (GPT-2 applies layer norm BEFORE attention, not after!)
        """
        # Tokenize
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding['input_ids']
        seq_len = input_ids.shape[1]
        
        # Get token embeddings (based on word identity)
        token_emb = self.model.transformer.wte(input_ids)  # [1, seq_len, d_model]
        
        # Get positional embeddings (based on position in sequence)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        pos_emb = self.model.transformer.wpe(pos_ids)  # [1, seq_len, d_model]
        
        # Combine: final embedding = token + position
        # This only happens at the INPUT layer!
        embeddings = (token_emb + pos_emb).squeeze(0)  # [seq_len, d_model]
        
        # Apply layer normalization if requested (GPT-2 does this before attention!)
        # Using layer 0's layer norm since these are input embeddings
        if apply_layer_norm:
            ln = self.model.transformer.h[0].ln_1  # Always use layer 0
            embeddings = ln(embeddings)
        
        return embeddings
    
 
    
    def predict_next_token(
        self,
        text: str,
        top_k: int = 10,
        show_probabilities: bool = True
    ) -> dict:
        """
        Predict the next token given input text and show top candidates.
        
        Args:
            text: Input text (prompt)
            top_k: Number of top predictions to show
            show_probabilities: Whether to display a probability bar chart
            
        Returns:
            Dictionary with predictions and probabilities
        """
        print(f"\n{'='*70}")
        print(f"NEXT TOKEN PREDICTION")
        print(f"{'='*70}")
        print(f"Input text: \"{text}\"")
        print(f"{'-'*70}")
        
        # Tokenize
        encoding = self.tokenizer(text, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get logits for the last token
        logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Decode tokens
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx])
            predictions.append({
                'token': token,
                'token_id': idx.item(),
                'probability': prob.item()
            })
        
        # Display results
        print(f"\nTop {top_k} predictions for next token:")
        print(f"{'Rank':<6} {'Token':<20} {'Token ID':<12} {'Probability'}")
        print(f"{'-'*55}")
        
        for i, pred in enumerate(predictions, 1):
            # Format token for display
            display_token = repr(pred['token']) if pred['token'].strip() != pred['token'] else pred['token']
            
            print(f"{i:<6} {display_token:<20} {pred['token_id']:<12} {pred['probability']:.4f}")
        
        return {
            'input_text': text,
            'predictions': predictions
        }
    
    def sample_next_token(self, text: str, temperature: float = 1.0, greedy: bool = False) -> tuple[int, str]:
        """
        Sample the next token given input text.
        
        This is a simple helper function for students to use when building
        their own autoregressive text generators.
        
        Args:
            text: Input text string
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
                        temperature=1.0 uses the raw probability distribution
                        temperature>1.0 makes distribution more uniform (more random)
                        temperature<1.0 makes distribution more peaked (less random)
                        Ignored when greedy=True
            greedy: If True, always select the highest probability token (deterministic)
                   If False, sample from the probability distribution (default)
            
        Returns:
            tuple of (token_id, token_string)
            - token_id: The integer ID of the sampled/selected token
            - token_string: The decoded string representation of the token
            
        Example:
            >>> # Sampling with temperature
            >>> token_id, token_str = lab.sample_next_token("The cat sat on the", temperature=0.7)
            >>> print(f"Sampled token ID: {token_id}, Token: '{token_str}'")
            >>> 
            >>> # Greedy selection (deterministic)
            >>> token_id, token_str = lab.sample_next_token("The cat sat on the", greedy=True)
            >>> print(f"Greedy token ID: {token_id}, Token: '{token_str}'")
        """
        # Tokenize input text
        encoding = self.tokenizer(text, return_tensors="pt")
        
        # Forward pass through model
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get logits for the last token
        logits = outputs.logits[0, -1, :]
        
        if greedy:
            # Select the token with highest probability (deterministic)
            next_token_id = torch.argmax(logits).item()
        else:
            # Apply temperature and sample from distribution
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # Decode the token to get its string representation
        next_token_str = self.tokenizer.decode([next_token_id])
        
        return next_token_id, next_token_str
    
    def interactive_completion(
        self,
        prompt: str,
        num_tokens: int = 20,
        temperature: float = 1.0,
        show_each_step: bool = False
    ) -> str:
        """
        Generate text by predicting tokens one at a time.
        
        Args:
            prompt: Starting text
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            show_each_step: Whether to show predictions at each step
            
        Returns:
            Generated text
        """
        print(f"\n{'='*70}")
        print(f"INTERACTIVE TEXT GENERATION")
        print(f"{'='*70}")
        print(f"Prompt: \"{prompt}\"")
        print(f"Generating {num_tokens} tokens with temperature={temperature}")
        print(f"{'-'*70}\n")
        
        current_text = prompt
        
        for i in range(num_tokens):
            # Tokenize current text
            encoding = self.tokenizer(current_text, return_tensors="pt")
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            # Get logits for the last token
            logits = outputs.logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Decode token
            next_token = self.tokenizer.decode([next_token_id])
            
            # Show step if requested
            if show_each_step:
                top_prob = probs[next_token_id].item()
                print(f"Step {i+1:2d}: Selected '{next_token}' "
                      f"(ID: {next_token_id}, P={top_prob:.4f})")
            
            # Append to current text
            current_text += next_token
        
        print(f"\n{'='*70}")
        print(f"GENERATED TEXT:")
        print(f"{'='*70}")
        print(f"{current_text}")
        print(f"{'='*70}")
        
        return current_text
    
    
    def analyze_embedding_dimensions(
        self,
        num_positions: int = 50,
        num_dims: int = 8,
        figsize: Tuple[int, int] = None
    ):
        """
        Analyze individual dimensions of positional embeddings.
        
        Args:
            num_positions: Number of positions to plot
            num_dims: Number of embedding dimensions to show
            figsize: Figure size for the plot (auto-calculated if None)
        """
        pos_emb = self.model.transformer.wpe.weight.data
        pos_emb_subset = pos_emb[:num_positions, :num_dims].cpu().numpy()
        
        # Dynamically calculate grid layout based on num_dims
        ncols = min(4, num_dims)  # Max 4 columns
        nrows = (num_dims + ncols - 1) // ncols  # Ceiling division
        
        # Auto-calculate figsize if not provided
        if figsize is None:
            figsize = (3.5 * ncols, 2 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Handle case where axes might not be 2D array
        if num_dims == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        positions = np.arange(num_positions)
        
        for i in range(num_dims):
            axes[i].plot(positions, pos_emb_subset[:, i], 
                        linewidth=2, color=f'C{i % 10}', alpha=0.8)
            axes[i].set_title(f'Dimension {i}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Position', fontsize=9)
            axes[i].set_ylabel('Value', fontsize=9)
            axes[i].grid(alpha=0.3)
            axes[i].axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Hide any extra subplots if num_dims doesn't fill the grid
        for i in range(num_dims, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle('Individual Positional Embedding Dimensions (LEARNED)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_word_embeddings(
        self,
        words: List[str],
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Visualize word embeddings using PCA to reduce to 2D.
        
        Shows how semantically similar words cluster together in embedding space.
        
        Args:
            words: List of words/tokens to visualize
            figsize: Figure size for the plot
            
        Note:
            - Uses token embeddings (not positional embeddings)
            - Words are tokenized with a leading space (as they appear in text)
            - Each word may be split into multiple tokens (e.g., " truck" → "truck")
            - Rare words may split into subwords (e.g., " supercalifragilistic" → multiple pieces)
            - PCA reduces from 768 dimensions to 2 for visualization
        """
        from sklearn.decomposition import PCA
        
        # Collect embeddings for each word
        embeddings_list = []
        labels = []
        
        for word in words:
            # Add space before word to match how it appears in text
            # GPT-2 tokenizes " truck" differently than "truck"
            word_with_space = ' ' + word
            tokens = self.tokenizer.tokenize(word_with_space)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Get embedding for each token
            for token, token_id in zip(tokens, token_ids):
                # Get token embedding (not position embedding!)
                emb = self.model.transformer.wte.weight[token_id].detach().cpu().numpy()
                embeddings_list.append(emb)
                # Clean up the Ġ character (represents space) for display
                clean_token = token.replace('Ġ', ' ').strip()
                labels.append(f"{word}→{clean_token}" if len(tokens) > 1 else word)
        
        # Stack into matrix [num_words, 768]
        embeddings_matrix = np.stack(embeddings_list)
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_matrix)
   
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           s=100, alpha=0.6, c=range(len(labels)), 
                           cmap='tab20')
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, 
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Word Embeddings in 2D (PCA Projection)',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
   
def main():
    """Main function demonstrating the lab activities."""
    
    # Initialize the lab
    lab = TransformerLab(model_name="gpt2")

    lab.visualize_word_embeddings(["cat", "dog", "car", "truck", "slow", "fast", "Paris", "Berlin"])
    


if __name__ == "__main__":
    main()
