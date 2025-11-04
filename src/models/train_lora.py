"""LoRA fine-tuning script for knowledge extraction"""

import os
from typing import Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from loguru import logger


def create_prompt(example: Dict[str, Any]) -> str:
    """Create training prompt from example"""
    instruction = """Extract knowledge graph triples from the following text.
Output format: S:subject|R:relation|O:object (one per line)"""

    prompt = f"""### Instruction:
{instruction}

### Input:
{example['input']}

### Output:
{example['output']}"""

    return prompt


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for training"""
    # Create prompts
    prompts = [create_prompt({'input': inp, 'output': out})
               for inp, out in zip(examples['input'], examples['output'])]

    # Tokenize
    result = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

    result['labels'] = result['input_ids'].clone()

    return result


def train_lora(
    config: Dict[str, Any],
    dataset_path: str,
    output_dir: str
) -> None:
    """Train LoRA adapter for knowledge extraction"""

    logger.info("Starting LoRA fine-tuning...")

    # Get configuration
    model_config = config.get('models', {})
    training_config = config.get('training', {})
    lora_config_dict = model_config.get('lora', {})

    base_model_name = model_config.get('base_model', 'meta-llama/Llama-2-7b-chat-hf')

    # Check if model path is available
    if 'BASE_MODEL_PATH' in os.environ:
        base_model_name = os.environ['BASE_MODEL_PATH']

    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('json', data_files={
        'train': f'{dataset_path}/train.json',
        'validation': f'{dataset_path}/val.json',
        'test': f'{dataset_path}/test.json'
    })

    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Val size: {len(dataset['validation'])}")
    logger.info(f"Test size: {len(dataset['test'])}")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    max_length = training_config.get('max_length', 512)

    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )

    # Load model
    logger.info("Loading base model...")

    # For demo purposes, we'll use fp16 if CUDA available, else fp32
    use_fp16 = training_config.get('fp16', False) and torch.cuda.is_available()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # Prepare model for training
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # Create LoRA config
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config_dict.get('r', 8),
        lora_alpha=lora_config_dict.get('alpha', 32),
        lora_dropout=lora_config_dict.get('dropout', 0.1),
        target_modules=lora_config_dict.get('target_modules', ['q_proj', 'v_proj'])
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    logger.info("Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get('batch_size', 4),
        per_device_eval_batch_size=training_config.get('batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 2e-4),
        num_train_epochs=training_config.get('num_epochs', 3),
        warmup_steps=training_config.get('warmup_steps', 100),
        logging_steps=training_config.get('logging_steps', 50),
        save_steps=training_config.get('save_steps', 500),
        eval_steps=training_config.get('eval_steps', 100),
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        fp16=use_fp16,
        report_to='none',  # Disable wandb/tensorboard for now
        save_total_limit=3,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    logger.info(f"Test results: {test_results}")

    logger.info("Training complete!")


class LoRAExtractor:
    """Extractor using fine-tuned LoRA model"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_path = config.get('models', {}).get('lora_model_path', './models/lora/extractor_v1/final')

        if 'LORA_MODEL_PATH' in os.environ:
            model_path = os.environ['LORA_MODEL_PATH']

        logger.info(f"Loading LoRA model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto' if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.model.eval()

    def extract_triples(self, text: str, source: str = None):
        """Extract triples using fine-tuned model"""
        from src.agents.extractor import Triple

        # Create prompt
        instruction = """Extract knowledge graph triples from the following text.
Output format: S:subject|R:relation|O:object (one per line)"""

        prompt = f"""### Instruction:
{instruction}

### Input:
{text}

### Output:
"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract output portion
        output_start = generated_text.find('### Output:')
        if output_start != -1:
            output_text = generated_text[output_start + len('### Output:'):].strip()
        else:
            output_text = generated_text

        # Parse triples
        triples = []
        for line in output_text.split('\n'):
            line = line.strip()
            if line and line.startswith('S:'):
                try:
                    parts = line.split('|')
                    subject = parts[0].replace('S:', '').strip()
                    relation = parts[1].replace('R:', '').strip()
                    obj = parts[2].replace('O:', '').strip()

                    triple = Triple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=0.9,
                        provenance={'source': source, 'method': 'lora'}
                    )
                    triples.append(triple)
                except Exception:
                    continue

        return triples
