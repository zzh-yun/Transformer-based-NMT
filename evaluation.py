#!/usr/bin/env python3
"""
Evaluation script for Neural Machine Translation model
Evaluates on test dataset and outputs BLEU-1, BLEU-2, BLEU-3, BLEU-4 and Perplexity scores
"""

import torch
import torch.nn as nn
import os
import jieba
import pickle
from typing import List, Optional
import time
import sentencepiece as spm
import numpy as np
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from src.model import TransformerNMT, BeamSearchDecoder, create_model
from src.dataset import DataPreprocessor, create_data_loaders
from src.utils import clean_text_zh
from collections import Counter


class BLEUEvaluator:
    """BLEU score evaluator for machine translation"""
    
    def __init__(self):
        pass
    
    def _get_ngrams(self, tokens, n):
        """Get n-grams from a list of tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def _get_precision(self, candidate_ngrams, reference_ngrams):
        """Calculate precision for n-grams"""
        matches = sum((candidate_ngrams & reference_ngrams).values())
        total = sum(candidate_ngrams.values())
        return matches / total if total > 0 else 0.0
    
    def _brevity_penalty(self, candidate_length, reference_length):
        """Calculate brevity penalty"""
        if candidate_length > reference_length:
            return 1.0
        return np.exp(1 - reference_length / candidate_length) if candidate_length > 0 else 0.0
    
    def sentence_bleu(self, reference, candidate, n_gram=4):
        """Calculate BLEU score for a single sentence"""
        reference_tokens = reference.split() if isinstance(reference, str) else reference
        candidate_tokens = candidate.split() if isinstance(candidate, str) else candidate
        
        if len(candidate_tokens) == 0:
            return 0.0
        
        # Calculate precision for each n-gram
        precisions = []
        for n in range(1, n_gram + 1):
            candidate_ngrams = self._get_ngrams(candidate_tokens, n)
            reference_ngrams = self._get_ngrams(reference_tokens, n)
            precision = self._get_precision(candidate_ngrams, reference_ngrams)
            precisions.append(precision)
        
        # Calculate geometric mean
        if min(precisions) == 0:
            return 0.0
        
        geometric_mean = np.exp(np.mean([np.log(p) for p in precisions if p > 0]))
        
        # Apply brevity penalty
        bp = self._brevity_penalty(len(candidate_tokens), len(reference_tokens))
        
        return bp * geometric_mean
    
    def corpus_bleu(self, references, candidates, n_gram=4):
        """Calculate corpus-level BLEU score"""
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")
        
        # Calculate precision for each n-gram level
        total_matches = [0] * n_gram
        total_candidate_ngrams = [0] * n_gram
        
        # For brevity penalty
        candidate_length = 0
        reference_length = 0
        
        for ref_list, cand in zip(references, candidates):
            # Handle multiple references (take the closest length)
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            ref_tokens_list = []
            for ref in ref_list:
                ref_tokens = ref.split() if isinstance(ref, str) else ref
                ref_tokens_list.append(ref_tokens)
            
            cand_tokens = cand.split() if isinstance(cand, str) else cand
            
            # Find closest reference length
            cand_len = len(cand_tokens)
            closest_ref = min(ref_tokens_list, key=lambda x: abs(len(x) - cand_len))
            
            candidate_length += cand_len
            reference_length += len(closest_ref)
            
            # Calculate matches for each n-gram
            for n in range(1, n_gram + 1):
                cand_ngrams = self._get_ngrams(cand_tokens, n)
                total_candidate_ngrams[n-1] += sum(cand_ngrams.values())
                
                # Find maximum matches across all references
                max_matches = 0
                for ref_tokens in ref_tokens_list:
                    ref_ngrams = self._get_ngrams(ref_tokens, n)
                    matches = sum((cand_ngrams & ref_ngrams).values())
                    max_matches = max(max_matches, matches)
                
                total_matches[n-1] += max_matches
        
        # Calculate precision for each n-gram
        precisions = []
        for n in range(n_gram):
            if total_candidate_ngrams[n] == 0:
                return 0.0
            precision = total_matches[n] / total_candidate_ngrams[n]
            precisions.append(precision)
        
        # Calculate geometric mean
        if min(precisions) == 0:
            return 0.0
        
        geometric_mean = np.exp(np.mean([np.log(p) for p in precisions if p > 0]))
        
        # Apply brevity penalty
        bp = self._brevity_penalty(candidate_length, reference_length)
        
        return bp * geometric_mean

# import debugpy
# debugpy.listen(17170)
# print("wait debugger")
# debugpy.wait_for_client()
# print('Debugger Attached')

class NMTInference:
    """Neural Machine Translation Inference Engine"""
    
    def __init__(self, model_path: str, config, device: str = 'auto'):
        """Initialize the inference engine"""
        self.device = self._setup_device(device)
        self.config = config
        
        # Load checkpoint first to get vocabularies
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        # Load vocabularies from checkpoint (preferred) or pickle files
        self.src_vocab, self.tgt_vocab = self._load_vocabularies(
            os.path.dirname(model_path), 
            checkpoint=checkpoint
        )
        
        self.model = self._load_model(model_path, checkpoint)
        
        self.beam_decoder = BeamSearchDecoder(
            self.model,
            beam_size=5,
            max_len=self.config.MAX_LEN,
            sos_idx=self.tgt_vocab.SOS_IDX,
            eos_idx=self.tgt_vocab.EOS_IDX,
            pad_idx=self.tgt_vocab.PAD_IDX
        )
        
        # Load SentencePiece model for decoding
        spm_model_path = os.path.join(os.path.dirname(model_path), 'en_spm_model.model')
        if not os.path.exists(spm_model_path):
            # Try root directory
            spm_model_path = 'en_spm_model.model'
        if os.path.exists(spm_model_path):
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(spm_model_path)
            print(f"SentencePiece model loaded from {spm_model_path}")
        else:
            self.sp_processor = None
            print("Warning: SentencePiece model not found, will use token-level decoding")
        
        # Initialize BLEU evaluator
        self.bleu_evaluator = BLEUEvaluator()
        
        # Loss function for perplexity calculation
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tgt_vocab.PAD_IDX,
            reduction='mean'
        )
        
        print(f"Model loaded on {self.device}")
        print(f"Source vocabulary size: {len(self.src_vocab)}")
        print(f"Target vocabulary size: {len(self.tgt_vocab)}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("Using CPU")
        
        return torch.device(device)

    def _load_vocabularies(self, path: str, checkpoint: dict = None) -> tuple:
        """Load source and target vocabularies from checkpoint or pickle files"""
        # Try to load from checkpoint first (preferred, ensures consistency)
        if checkpoint and 'src_vocab' in checkpoint and 'tgt_vocab' in checkpoint:
            print("Loading vocabularies from checkpoint...")
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
        else:
            # Fall back to loading from pickle files
            print("Loading vocabularies from pickle files...")
            src_vocab_path = os.path.join(path, 'src_vocab.pkl')
            tgt_vocab_path = os.path.join(path, 'tgt_vocab.pkl')
            
            if not os.path.exists(src_vocab_path) or not os.path.exists(tgt_vocab_path):
                raise FileNotFoundError("Vocabulary files (src_vocab.pkl, tgt_vocab.pkl) not found.")
                
            with open(src_vocab_path, 'rb') as f:
                src_vocab = pickle.load(f)
            with open(tgt_vocab_path, 'rb') as f:
                tgt_vocab = pickle.load(f)
        
        # Debug: Check vocabulary integrity
        print(f"Source vocab size: {len(src_vocab)}")
        print(f"Source vocab idx2word size: {len(src_vocab.idx2word)}")
        print(f"Target vocab size: {len(tgt_vocab)}")
        print(f"Target vocab idx2word size: {len(tgt_vocab.idx2word)}")
        
        # Check if idx2word is complete
        max_src_idx = max(src_vocab.idx2word.keys()) if src_vocab.idx2word else -1
        max_tgt_idx = max(tgt_vocab.idx2word.keys()) if tgt_vocab.idx2word else -1
        print(f"Source vocab max index: {max_src_idx}, vocab size: {len(src_vocab)}")
        print(f"Target vocab max index: {max_tgt_idx}, vocab size: {len(tgt_vocab)}")
        
        # Check for missing indices
        if max_src_idx >= len(src_vocab):
            missing_src = [i for i in range(len(src_vocab)) if i not in src_vocab.idx2word]
            print(f"Warning: Source vocab has {len(missing_src)} missing indices: {missing_src[:10]}")
        
        if max_tgt_idx >= len(tgt_vocab):
            missing_tgt = [i for i in range(len(tgt_vocab)) if i not in tgt_vocab.idx2word]
            print(f"Warning: Target vocab has {len(missing_tgt)} missing indices: {missing_tgt[:10]}")
            
        return src_vocab, tgt_vocab

    def _load_model(self, model_path: str, checkpoint: dict = None) -> TransformerNMT:
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['model_state_dict']
        
        # Infer model parameters from checkpoint state_dict
        # This ensures we use the same vocabulary sizes and max_len as training
        src_vocab_size = state_dict['src_embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['tgt_embedding.weight'].shape[0]
        
        # Get max_len from pos_encoder or relative_pos_encoding
        if 'pos_encoder.pe' in state_dict:
            max_len = state_dict['pos_encoder.pe'].shape[0]
        elif 'relative_pos_encoding.relative_position_bias_table' in state_dict:
            # For relative position encoding, max_len can be inferred from bias table
            # bias table shape: (2 * max_len - 1, num_heads)
            bias_table_shape = state_dict['relative_pos_encoding.relative_position_bias_table'].shape
            max_len = (bias_table_shape[0] + 1) // 2
        else:
            # Fallback to config
            max_len = getattr(self.config, 'MAX_LEN', 128)
        
        # Infer other model parameters from state_dict or use config defaults
        d_model = state_dict['src_embedding.weight'].shape[1]
        
        # Get pos_encoding_type and norm_type from checkpoint if available
        pos_encoding_type = checkpoint.get('pos_encoding_type', getattr(self.config, 'POS_ENCODING_TYPE', 'absolute'))
        norm_type = checkpoint.get('norm_type', getattr(self.config, 'NORM_TYPE', 'layernorm'))
        
        # If not in checkpoint, try to infer from state_dict keys
        # Custom layers use "transformer_encoder.0" while standard layers use "transformer_encoder.layers.0"
        if 'pos_encoding_type' not in checkpoint and 'norm_type' not in checkpoint:
            has_custom_layers = any(key.startswith('transformer_encoder.0.') for key in state_dict.keys())
            if has_custom_layers:
                # Check if it's RMSNorm (no bias in norm layers) or relative pos encoding
                has_rmsnorm = any('norm1.weight' in key and 'norm1.bias' not in key 
                                 for key in state_dict.keys() if 'transformer_encoder.0.norm' in key)
                has_relative_pos = 'relative_pos_encoding.relative_position_bias_table' in state_dict
                
                if has_relative_pos:
                    pos_encoding_type = 'relative'
                if has_rmsnorm:
                    norm_type = 'rmsnorm'
        
        print(f"Inferred from checkpoint:")
        print(f"  Source vocab size: {src_vocab_size}")
        print(f"  Target vocab size: {tgt_vocab_size}")
        print(f"  Max length: {max_len}")
        print(f"  d_model: {d_model}")
        print(f"  Position encoding type: {pos_encoding_type}")
        print(f"  Normalization type: {norm_type}")
        
        # Verify vocabulary sizes match
        if len(self.src_vocab) != src_vocab_size:
            print(f"Warning: Source vocab size mismatch! Checkpoint: {src_vocab_size}, Loaded: {len(self.src_vocab)}")
            print("Using checkpoint vocabulary size for model creation.")
        if len(self.tgt_vocab) != tgt_vocab_size:
            print(f"Warning: Target vocab size mismatch! Checkpoint: {tgt_vocab_size}, Loaded: {len(self.tgt_vocab)}")
            print("Using checkpoint vocabulary size for model creation.")
        
        # Create model using inferred parameters
        model = create_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=self.device,
            pos_encoding_type=pos_encoding_type,
            norm_type=norm_type,
            d_model=d_model,
            nhead=self.config.NHEAD,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.NUM_DECODER_LAYERS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            dropout=self.config.DROPOUT,
            max_len=max_len
        )
        
        # Load model state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        # Update config MAX_LEN to match checkpoint
        self.config.MAX_LEN = max_len
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"Best BLEU score: {checkpoint.get('bleu_score', 'N/A')}")
        
        return model
    
    def preprocess_text(self, text: str, is_chinese: bool = True) -> str:
        """Preprocess input text"""
        # Basic text cleaning
        text = text.strip()
        
        if is_chinese:
            # For Chinese text, assume it's already segmented
            # In practice, you might want to use jieba or other segmentation tools
            pass
        else:
            # For English text, convert to lowercase
            text = text.lower()
        
        return text
    
    def translate_sentence(self, sentence: str, use_beam_search: bool = True, verbose: bool = False) -> str:
        """Translate a single sentence"""
        # Preprocess
        clean_data = clean_text_zh(sentence)
        src_tokens = list(jieba.cut(clean_data, cut_all=False))
        
        if verbose:
            print(f"Debug - Input sentence: {sentence}")
            print(f"Debug - Cleaned text: {clean_data}")
            print(f"Debug - Tokenized tokens: {src_tokens}")
        
        # Convert to indices
        src_indices = self.src_vocab.tokens_to_indices(src_tokens)
        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device).unsqueeze(1)
        
        with torch.no_grad():
            if use_beam_search:
                # Use beam search decoding
                src_key_padding_mask = self.model.create_padding_mask(src_tensor, self.src_vocab.PAD_IDX)
                results = self.beam_decoder.decode(src_tensor, src_key_padding_mask=src_key_padding_mask)
                
                if results:
                    translated_indices = results[0].cpu().tolist()
                    translated_tokens = self.tgt_vocab.indices_to_tokens(translated_indices)
                    
                    # Use SentencePiece decode if available
                    if self.sp_processor is not None:
                        translated_text = self.sp_processor.decode(translated_tokens)
                    else:
                        translated_text = " ".join(translated_tokens)
                else:
                    translated_text = ""
            else:
                # Use greedy decoding
                translated_text = self._greedy_decode(src_tensor, verbose=verbose)
        
        return translated_text
    
    def _greedy_decode(self, src: torch.Tensor, verbose: bool = False) -> str:
        """Greedy decoding for single sentence"""
        batch_size = src.size(1)
        max_len = self.config.MAX_LEN
        
        # Initialize decoder input
        decoder_input = torch.full((1, batch_size), self.tgt_vocab.SOS_IDX, dtype=torch.long, device=self.device)
        
        for step in range(max_len):
            # Create masks
            tgt_mask = self.model.generate_square_subsequent_mask(decoder_input.size(0)).to(self.device)
            src_key_padding_mask = self.model.create_padding_mask(src, self.src_vocab.PAD_IDX)
            tgt_key_padding_mask = self.model.create_padding_mask(decoder_input, self.tgt_vocab.PAD_IDX)
            
            # Forward pass
            output = self.model(
                src=src,
                tgt=decoder_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Get next token
            next_token = torch.argmax(output[-1, :, :], dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=0)
            
            # Check for EOS
            if next_token.item() == self.tgt_vocab.EOS_IDX:
                break
        
        # Convert to text
        translated_indices = decoder_input[1:, 0].cpu().tolist()  # Skip SOS token
        translated_tokens = self.tgt_vocab.indices_to_tokens(translated_indices)
        
        # Use SentencePiece decode if available
        if self.sp_processor is not None:
            translated_text = self.sp_processor.decode(translated_tokens)
        else:
            translated_text = " ".join(translated_tokens)
        
        return translated_text
    
    def translate_batch(self, sentences: List[str], use_beam_search: bool = True) -> List[str]:
        """Translate a batch of sentences"""
        results = []
        
        for sentence in sentences:
            translated = self.translate_sentence(sentence, use_beam_search)
            results.append(translated)
        
        return results
    
    def evaluate_on_test_set(self, data_dir: str = None, use_beam_search: bool = True) -> dict:
        """Evaluate model on test dataset using DataLoader"""
        print("=" * 60)
        print("Evaluating on Test Dataset")
        print("=" * 60)
        
        # Load test dataset using DataPreprocessor
        preprocessor = DataPreprocessor(self.config)
        
        # Use vocabularies already loaded from checkpoint (preferred)
        if hasattr(self, 'src_vocab') and hasattr(self, 'tgt_vocab'):
            print("Using vocabularies from loaded checkpoint...")
            preprocessor.src_vocab = self.src_vocab
            preprocessor.tgt_vocab = self.tgt_vocab
        else:
            # Fallback: Try to load vocabularies from files
            model_dir = os.path.dirname(self.config.model_path)
            vocab_loaded = False
            
            # Try 1: From model directory (experiment-specific directory)
            if preprocessor.load_vocabularies(model_dir):
                vocab_loaded = True
            else:
                # Try 2: From base checkpoint directory
                base_checkpoint_dir = getattr(self.config, 'CHECKPOINT_DIR', 'checkpoints')
                if preprocessor.load_vocabularies(base_checkpoint_dir):
                    vocab_loaded = True
                else:
                    # Try 3: From parent directory of model path
                    parent_dir = os.path.dirname(model_dir)
                    if parent_dir and preprocessor.load_vocabularies(parent_dir):
                        vocab_loaded = True
            
            if not vocab_loaded:
                raise FileNotFoundError(
                    "Vocabulary files not found. Please ensure vocabularies are saved.\n"
                    f"Tried locations: {model_dir}, {base_checkpoint_dir}"
                )
        
        # Create test dataset (vocabularies are already set, so create_datasets will use them)
        # We need to load the data but not rebuild vocabularies
        from src.utils import get_data
        data_dir = data_dir or self.config.DATA_DIR
        (src_train_sents, tgt_train_sents,
         src_valid_sents, tgt_valid_sents,
         src_test_sents, tgt_test_sents) = get_data(data_dir)
        
        # Create test dataset directly without rebuilding vocabularies
        from src.dataset import NMTDataset
        test_dataset = NMTDataset(
            src_test_sents, tgt_test_sents, 
            preprocessor.src_vocab, preprocessor.tgt_vocab, 
            max_len=self.config.MAX_LEN
        )
        
        # Create test loader directly with custom collate function
        from torch.utils.data import DataLoader
        
        def test_collate_fn(batch):
            """Custom collate function for test data"""
            src_batch, tgt_input_batch, tgt_output_batch = [], [], []
            pad_idx = self.tgt_vocab.PAD_IDX  # Use target vocab PAD_IDX
            
            for item in batch:
                src_batch.append(item['src'])
                tgt_input_batch.append(item['tgt_input'])
                tgt_output_batch.append(item['tgt_output'])
            
            src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx, batch_first=False)
            tgt_input_batch = torch.nn.utils.rnn.pad_sequence(tgt_input_batch, padding_value=pad_idx, batch_first=False)
            tgt_output_batch = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, padding_value=pad_idx, batch_first=False)
            
            return {
                'src': src_batch,
                'tgt_input': tgt_input_batch,
                'tgt_output': tgt_output_batch
            }
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('BATCH_SIZE', 32),
            shuffle=False,
            num_workers=self.config.get('NUM_WORKERS', 4),
            collate_fn=test_collate_fn,
            pin_memory=True
        )
        
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Evaluation
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            
            for batch in pbar:
                # Move to device
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Calculate loss (for perplexity)
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
                src_key_padding_mask = self.model.create_padding_mask(src, self.src_vocab.PAD_IDX).float()
                tgt_key_padding_mask = self.model.create_padding_mask(tgt_input, self.tgt_vocab.PAD_IDX).float()
                
                # Forward pass for loss calculation
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                # Get predictions using autoregressive decoding
                batch_size = src.size(1)
                max_len = min(self.config.MAX_LEN, getattr(self.config, 'VAL_MAX_LEN', 64))
                
                # Initialize decoder input
                decoder_input = torch.full((1, batch_size), self.tgt_vocab.SOS_IDX, 
                                         dtype=torch.long, device=self.device)
                
                # Autoregressive decoding
                for step in range(max_len):
                    tgt_mask_dec = self.model.generate_square_subsequent_mask(decoder_input.size(0)).to(self.device)
                    tgt_key_padding_mask_dec = self.model.create_padding_mask(decoder_input, self.tgt_vocab.PAD_IDX).float()
                    
                    output_dec = self.model(
                        src=src,
                        tgt=decoder_input,
                        tgt_mask=tgt_mask_dec,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask_dec,
                        memory_key_padding_mask=src_key_padding_mask
                    )
                    
                    # Get next token
                    next_token = torch.argmax(output_dec[-1], dim=-1)  # (batch_size,)
                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=0)
                    
                    # Check for EOS
                    if torch.all(next_token == self.tgt_vocab.EOS_IDX):
                        break
                
                # Convert to text
                pred = decoder_input[1:].transpose(0, 1)  # (batch_size, seq_len)
                pred_text = preprocessor.tgt_vocab.decode_batch(pred.cpu().numpy())
                ref_text = preprocessor.tgt_vocab.decode_batch(batch['tgt_output'].transpose(0,1).cpu().numpy())
                
                all_predictions.extend(pred_text)
                all_references.extend([[r] for r in ref_text])
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / len(test_loader)
        perplexity = np.exp(avg_loss)
        
        # Calculate BLEU scores
        bleu_1 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=1)
        bleu_2 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=2)
        bleu_3 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=3)
        bleu_4 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=4)
        
        results = {
            'num_sentences': len(all_predictions),
            'loss': avg_loss,
            'perplexity': perplexity,
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("Evaluation Results on Test Dataset")
        print("=" * 60)
        print(f"Number of sentences: {results['num_sentences']}")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"BLEU-1: {results['bleu_1']:.4f}")
        print(f"BLEU-2: {results['bleu_2']:.4f}")
        print(f"BLEU-3: {results['bleu_3']:.4f}")
        print(f"BLEU-4: {results['bleu_4']:.4f}")
        print("=" * 60)
        
        return results
    
    def interactive_translation(self):
        """Interactive translation mode"""
        print("=" * 60)
        print("INTERACTIVE TRANSLATION MODE")
        print("Enter Chinese sentences to translate to English")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            try:
                # Get input
                chinese_text = input("\n中文输入: ").strip()
                
                if chinese_text.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not chinese_text:
                    continue
                
                # Translate
                start_time = time.time()
                english_text = self.translate_sentence(chinese_text)
                translation_time = time.time() - start_time
                
                # Display result
                print(f"English: {english_text}")
                print(f"Time: {translation_time:.3f}s")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Translation error: {e}")
        
        print("\nGoodbye!")

@hydra.main(version_base='1.3', config_path='./configs', config_name='inference.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    """Main evaluation function - evaluates on test dataset"""
    try:
        # Add model_path to config if not present
        if not hasattr(cfgs, 'model_path') or cfgs.model_path is None:
            raise ValueError("model_path must be specified in config or command line")
        
        inference_engine = NMTInference(cfgs.model_path, cfgs, cfgs.device)
        
        # Evaluate on test dataset
        results = inference_engine.evaluate_on_test_set(
            data_dir=cfgs.get('DATA_DIR'),
            use_beam_search=cfgs.get('use_beam_search', True)
        )
        
        return results['bleu_4']  # Return BLEU-4 as main metric

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model path and vocabulary files are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 