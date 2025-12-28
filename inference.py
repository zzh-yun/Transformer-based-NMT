#!/usr/bin/env python3
"""
Simplified evaluation script for Neural Machine Translation model
Only outputs input and translated text (two lines)
"""

import torch
import os
import jieba
import pickle
import sentencepiece as spm
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from src.model import TransformerNMT, BeamSearchDecoder, create_model
from src.utils import clean_text_zh


class SimpleNMTInference:
    """Simplified NMT Inference Engine - no debug output"""
    
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
            spm_model_path = 'en_spm_model.model'
        if os.path.exists(spm_model_path):
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(spm_model_path)
        else:
            self.sp_processor = None
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return torch.device(device)

    def _load_vocabularies(self, path: str, checkpoint: dict = None) -> tuple:
        """Load source and target vocabularies from checkpoint or pickle files"""
        if checkpoint and 'src_vocab' in checkpoint and 'tgt_vocab' in checkpoint:
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
        else:
            src_vocab_path = os.path.join(path, 'src_vocab.pkl')
            tgt_vocab_path = os.path.join(path, 'tgt_vocab.pkl')
            
            if not os.path.exists(src_vocab_path) or not os.path.exists(tgt_vocab_path):
                raise FileNotFoundError("Vocabulary files (src_vocab.pkl, tgt_vocab.pkl) not found.")
                
            with open(src_vocab_path, 'rb') as f:
                src_vocab = pickle.load(f)
            with open(tgt_vocab_path, 'rb') as f:
                tgt_vocab = pickle.load(f)
            
        return src_vocab, tgt_vocab

    def _load_model(self, model_path: str, checkpoint: dict = None) -> TransformerNMT:
        """Load trained model"""
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['model_state_dict']
        
        # Infer model parameters from checkpoint state_dict
        src_vocab_size = state_dict['src_embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['tgt_embedding.weight'].shape[0]
        max_len = state_dict['pos_encoder.pe'].shape[0]
        d_model = state_dict['src_embedding.weight'].shape[1]
        
        # Create model using inferred parameters
        model = create_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=self.device,
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
        
        return model
    
    def translate_sentence(self, sentence: str, use_beam_search: bool = True) -> str:
        """Translate a single sentence - no debug output"""
        # Preprocess
        clean_data = clean_text_zh(sentence)
        src_tokens = list(jieba.cut(clean_data, cut_all=False))
        
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
                translated_text = self._greedy_decode(src_tensor)
        
        return translated_text
    
    def _greedy_decode(self, src: torch.Tensor) -> str:
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


@hydra.main(version_base='1.3', config_path='./configs', config_name='inference.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    """Main evaluation function - only outputs input and translation"""
    try:
        inference_engine = SimpleNMTInference(cfgs.model_path, cfgs, cfgs.device)
        
        if cfgs.input_text:
            translation = inference_engine.translate_sentence(cfgs.input_text)
            print(f"Input:    {cfgs.input_text}")
            print(f"Translated: {translation}")
        elif cfgs.input_file:
            with open(cfgs.input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            
            for sentence in sentences:
                translation = inference_engine.translate_sentence(sentence)
                print(f"Input:    {sentence}")
                print(f"Translated: {translation}")
        else:
            print("Error: Please specify --input_text or --input_file")
            return None

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model path and vocabulary files are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

