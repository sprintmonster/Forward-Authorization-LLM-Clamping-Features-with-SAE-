import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from sae_lens import SAE
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.utils import ModelOutput


MODEL_ID = "google/gemma-3-1b-it"

# 주의:
# gemma-scope-2-1b-pt-res-all은 모든 resid_post 위치를 포함하는
# Gemma 3 1B pretrained residual SAE release입니다.
# Gemma 3 1B IT와 hidden size는 같더라도 activation distribution은 다를 수 있습니다.
SAE_RELEASE = "gemma-scope-2-1b-pt-res-all"
SAE_ID = "layer_22_width_16k_l0_big"
SAE_ID_TEMPLATE = "layer_{layer_index}_width_16k_l0_big"
LAYER_GROUP_SIZE = 5

LAYER_INDEX = 22
DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "trainset.csv"


def get_sae_id(layer_index: int) -> str:
    return SAE_ID_TEMPLATE.format(layer_index=layer_index)


def make_layer_groups(layer_indices, group_size: int = LAYER_GROUP_SIZE):
    if group_size < 1:
        raise ValueError("group_size must be at least 1.")

    layer_indices = list(layer_indices)
    return [
        layer_indices[start : start + group_size]
        for start in range(0, len(layer_indices), group_size)
    ]

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
    )


class AnswerDataset(Dataset):
    """
    trainset.csv에 다음 컬럼이 필요합니다.

    - Question

    SAE는 answer가 아닌 prompt activation만 학습합니다.
    """

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.items = []

        for row in frame.itertuples(index=False):
            prompt = make_prompt(
                question=str(row.Question),
            )
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
            ).input_ids

            input_ids = prompt_ids[:max_length]

            if not input_ids:
                continue

            self.items.append(
                {
                    "input_ids": input_ids,
                }
            )

        if not self.items:
            raise ValueError(
                "Dataset is empty after tokenization. "
                "Increase --max-length or check trainset.csv."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@dataclass
class CausalDataCollator:
    tokenizer: Any

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_mask = []

        for item in features:
            padding_length = max_length - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [self.tokenizer.pad_token_id] * padding_length
            )
            attention_mask.append(
                [1] * len(item["input_ids"])
                + [0] * padding_length
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


def split_data(data_path: Path, output_dir: Path, seed: int):
    frame = pd.read_csv(data_path)

    required_columns = {"Question"}
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    frame = frame.dropna(subset=list(required_columns)).reset_index(drop=True)

    if len(frame) < 2:
        raise ValueError("At least 2 valid rows are required.")

    validation = frame.sample(frac=0.1, random_state=seed)

    # 데이터가 아주 작아 validation이 0개가 되는 것을 방지
    if len(validation) == 0:
        validation = frame.sample(n=1, random_state=seed)

    train = (
        frame.drop(validation.index)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    validation = validation.reset_index(drop=True)

    if len(train) == 0:
        raise ValueError("Train split is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train_split.csv", index=False)
    validation.to_csv(output_dir / "validation_split.csv", index=False)

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

    return train, validation


def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer, device


def get_layer(gemma_model, layer_index: int):
    """
    Gemma 계열 모델의 decoder layer container를 찾습니다.

    보통:
    model.model.layers[layer_index]

    구조가 달라진 경우를 대비해 명시적으로 오류를 냅니다.
    """
    if not hasattr(gemma_model, "model"):
        raise AttributeError(
            "Cannot find gemma_model.model. "
            "Check the model architecture."
        )

    if not hasattr(gemma_model.model, "layers"):
        raise AttributeError(
            "Cannot find gemma_model.model.layers. "
            "Check the model architecture."
        )

    layers = gemma_model.model.layers

    if not 0 <= layer_index < len(layers):
        raise IndexError(
            f"LAYER_INDEX={layer_index} is invalid. "
            f"Model has {len(layers)} layers."
        )

    return layers[layer_index]


@dataclass
class SAEOutput(ModelOutput):
    loss: Any = None
    logits: Any = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None
    sae_reconstruction_loss: Any = None
    sae_sparsity_loss: Any = None
    sae_l0: Any = None


class FrozenGemmaSAE(nn.Module):
    """
    Gemma는 동결하고, 지정 레이어의 출력에 SAE를 삽입합니다.

    Hook 흐름:
        original hidden state h
            -> SAE encode -> z
            -> SAE decode -> h_hat
            -> 이후 Gemma layers에 h_hat 전달

    저장하는 값:
        target: original h, detached
        reconstructed: h_hat
        latent: z

    최종 loss는 Trainer.compute_loss에서 계산합니다.
    """

    def __init__(
        self,
        gemma,
        sae,
        layer_index: int,
        recon_loss_weight: float,
        sparsity_loss_weight: float,
    ):
        super().__init__()

        self.gemma = gemma
        self.sae = sae
        self.layer_index = layer_index

        self.recon_loss_weight = recon_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight

        # Gemma의 모든 파라미터는 업데이트 금지
        self.gemma.requires_grad_(False)

        # SAE 파라미터만 업데이트 허용
        self.sae.requires_grad_(True)

        self._sae_stats = None

        target_layer = get_layer(self.gemma, self.layer_index)

        self._hook = target_layer.register_forward_hook(
            self._reconstruct_residual
        )

    def train(self, mode: bool = True):
        """
        Trainer가 model.train()을 호출해도:
        - Gemma: 항상 eval mode
        - SAE: train/eval mode 전환

        Gemma의 dropout 등 stochastic layer를 고정하기 위함입니다.
        """
        super().train(mode)
        self.gemma.eval()
        self.sae.train(mode)
        return self

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _reconstruct_residual(self, _module, _inputs, output):
        """
        Transformer block output 구조는 버전에 따라 tensor 또는 tuple일 수 있습니다.
        첫 원소를 hidden state로 처리합니다.
        """
        hidden_states = output[0] if isinstance(output, tuple) else output

        # Gemma는 freeze 상태이므로 target activation에는 gradient가 필요 없습니다.
        target_hidden_states = hidden_states.detach()

        # SAE만 gradient를 받습니다.
        latent = self.sae.encode(target_hidden_states)
        reconstructed = self.sae.decode(latent)

        self._sae_stats = {
            "target": target_hidden_states,
            "reconstructed": reconstructed,
            "latent": latent,
        }

        # 이후 Gemma 레이어가 reconstructed activation을 받도록 치환
        if isinstance(output, tuple):
            return (reconstructed, *output[1:])

        return reconstructed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **inputs,
    ):
        self._sae_stats = None

        outputs = self.gemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **inputs,
        )

        if self._sae_stats is None:
            raise RuntimeError(
                "SAE hook did not run. "
                f"Check LAYER_INDEX={self.layer_index}."
            )

        target = self._sae_stats["target"]
        reconstructed = self._sae_stats["reconstructed"]
        latent = self._sae_stats["latent"]

        # target shape: (batch, sequence, d_model)
        # attention_mask shape: (batch, sequence)
        if attention_mask is None:
            token_mask = torch.ones(
                target.shape[:2],
                dtype=torch.bool,
                device=target.device,
            )
        else:
            token_mask = attention_mask.bool()

        valid_target = target[token_mask]
        valid_reconstructed = reconstructed[token_mask]
        valid_latent = latent[token_mask]

        if valid_target.numel() == 0:
            raise RuntimeError("No valid tokens remain after attention masking.")

        # SAE reconstruction objective:
        # h와 h_hat의 차이를 줄임
        reconstruction_loss = F.mse_loss(
            valid_reconstructed.float(),
            valid_target.float(),
        )

        # SAE sparsity objective:
        # latent feature activation의 평균 L1 값을 줄임
        sparsity_loss = valid_latent.float().abs().mean()

        # 활성 feature 수. loss에는 넣지 않고 모니터링 용도.
        l0 = (valid_latent > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sae_reconstruction_loss=reconstruction_loss,
            sae_sparsity_loss=sparsity_loss,
            sae_l0=l0,
        )

    def save_sae_checkpoint(self, path: Path):
        """
        Gemma는 변경되지 않으므로 SAE weight와 학습 설정만 저장합니다.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "sae_state_dict": self.sae.state_dict(),
                "model_id": MODEL_ID,
                "sae_release": SAE_RELEASE,
                "sae_id": get_sae_id(self.layer_index),
                "layer_index": self.layer_index,
                "recon_loss_weight": self.recon_loss_weight,
                "sparsity_loss_weight": self.sparsity_loss_weight,
            },
            path,
        )

        print(f"Saved SAE checkpoint: {path}")


class SAETrainer(Trainer):
    """
    최종 loss:
        total =
            recon_weight * SAE_reconstruction_loss
            + sparse_weight * SAE_L1_loss

    optimizer에는 model.sae.parameters()만 들어갑니다.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)

        reconstruction_loss = outputs.sae_reconstruction_loss
        sparsity_loss = outputs.sae_sparsity_loss

        total_loss = (
            model.recon_loss_weight * reconstruction_loss
            + model.sparsity_loss_weight * sparsity_loss
        )

        self.log(
            {
                "total_loss": total_loss.detach().float().item(),
                "sae_reconstruction_loss": (
                    reconstruction_loss.detach().float().item()
                ),
                "sae_sparsity_loss": (
                    sparsity_loss.detach().float().item()
                ),
                "sae_l0": outputs.sae_l0.detach().float().item(),
            }
        )

        if return_outputs:
            return total_loss, outputs

        return total_loss

    def create_optimizer(self):
        """
        명시적으로 SAE parameter만 optimizer에 넣습니다.

        따라서 Gemma는:
        1. requires_grad=False
        2. optimizer parameter group에 없음

        두 가지 이유로 절대 업데이트되지 않습니다.
        """
        if self.optimizer is None:
            sae_parameters = [
                parameter
                for parameter in self.model.sae.parameters()
                if parameter.requires_grad
            ]

            if not sae_parameters:
                raise RuntimeError("No trainable SAE parameters found.")

            self.optimizer = torch.optim.AdamW(
                sae_parameters,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        """
        Trainer checkpoint 저장 때 대형 Gemma 전체 weight 저장을 피하고,
        SAE weight만 저장합니다.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_path = Path(output_dir) / "sae_only.pt"

        self.model.save_sae_checkpoint(checkpoint_path)


def save_loss_history(trainer, output_dir: Path):
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(output_dir / "loss_history.csv", index=False)

    if history.empty:
        print("No loss values were recorded.")
        return

    plt.figure(figsize=(10, 6))

    metric_columns = [
        ("total_loss", "Total loss"),
        ("sae_reconstruction_loss", "SAE reconstruction loss"),
        ("sae_sparsity_loss", "SAE sparsity loss"),
    ]

    plotted = False

    for column, label in metric_columns:
        if column in history.columns:
            metric_history = history.dropna(subset=[column])

            if not metric_history.empty and "step" in metric_history.columns:
                plt.plot(
                    metric_history["step"],
                    metric_history[column],
                    label=label,
                )
                plotted = True

    if "eval_loss" in history.columns:
        eval_history = history.dropna(subset=["eval_loss"])

        if not eval_history.empty and "step" in eval_history.columns:
            plt.plot(
                eval_history["step"],
                eval_history["eval_loss"],
                "o-",
                label="Evaluation total loss",
            )
            plotted = True

    if not plotted:
        plt.close()
        print("No plottable loss metrics were recorded.")
        return

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Frozen Gemma + Trainable SAE Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure_path = output_dir / "loss_curve.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()

    print(f"Saved loss graph: {figure_path}")


def verify_parameter_policy(model: FrozenGemmaSAE):
    gemma_trainable = [
        name
        for name, parameter in model.gemma.named_parameters()
        if parameter.requires_grad
    ]

    sae_trainable = [
        name
        for name, parameter in model.sae.named_parameters()
        if parameter.requires_grad
    ]

    if gemma_trainable:
        raise RuntimeError(
            "Gemma has trainable parameters, but it must be frozen. "
            f"Examples: {gemma_trainable[:5]}"
        )

    if not sae_trainable:
        raise RuntimeError("SAE has no trainable parameters.")

    gemma_total = sum(
        parameter.numel()
        for parameter in model.gemma.parameters()
    )

    sae_total = sum(
        parameter.numel()
        for parameter in model.sae.parameters()
    )

    sae_trainable_total = sum(
        parameter.numel()
        for parameter in model.sae.parameters()
        if parameter.requires_grad
    )

    print(f"Gemma parameters (frozen): {gemma_total:,}")
    print(f"SAE parameters (total): {sae_total:,}")
    print(f"SAE parameters (trainable): {sae_trainable_total:,}")
    print("Update policy: SAE only")


def validate_model_sae_compatibility(gemma, sae):
    gemma_hidden_size = gemma.config.hidden_size
    sae_input_dim = sae.cfg.d_in

    print(f"Gemma hidden size: {gemma_hidden_size}")
    print(f"SAE d_in: {sae_input_dim}")

    if hasattr(sae.cfg, "metadata"):
        print(f"SAE hook name: {sae.cfg.metadata.hook_name}")
        print(f"SAE HF hook: {sae.cfg.metadata.hf_hook_name}")

    if gemma_hidden_size != sae_input_dim:
        raise ValueError(
            "Gemma hidden size and SAE input dimension do not match: "
            f"{gemma_hidden_size} != {sae_input_dim}"
        )


def train_layer(
    args,
    gemma,
    tokenizer,
    device: str,
    train_dataset,
    validation_dataset,
    layer_index: int,
    layer_output_dir: Path,
):
    sae_id = get_sae_id(layer_index)
    print(f"\n=== Training layer {layer_index}: {sae_id} ===")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=device,
    )
    validate_model_sae_compatibility(gemma, sae)

    model = FrozenGemmaSAE(
        gemma=gemma,
        sae=sae,
        layer_index=layer_index,
        recon_loss_weight=args.recon_loss_weight,
        sparsity_loss_weight=args.sparsity_loss_weight,
    )
    model.to(device)
    verify_parameter_policy(model)

    training_args = TrainingArguments(
        output_dir=str(layer_output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        weight_decay=0.0,
        dataloader_pin_memory=device == "cuda",
    )

    trainer = SAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalDataCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    try:
        trainer.train()
        save_loss_history(trainer, layer_output_dir)
        model.save_sae_checkpoint(layer_output_dir / "final_sae.pt")
        model.gemma.config.save_pretrained(layer_output_dir / "gemma_config")
        tokenizer.save_pretrained(layer_output_dir / "tokenizer")
    finally:
        model.remove_hook()
        del trainer, model, sae
        if device == "cuda":
            torch.cuda.empty_cache()


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame = split_data(
        data_path=DATA_PATH,
        output_dir=output_dir,
        seed=args.seed,
    )

    gemma, tokenizer, device = load_model_and_tokenizer()
    gemma.to(device).eval()
    train_dataset = AnswerDataset(
        frame=train_frame,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    validation_dataset = AnswerDataset(
        frame=validation_frame,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if args.layer_index is None:
        layer_indices = list(range(len(gemma.model.layers)))
    else:
        layer_indices = [args.layer_index]

    layer_groups = make_layer_groups(
        layer_indices,
        group_size=args.layer_group_size,
    )
    print(f"Layer groups: {layer_groups}")

    for group_index, layer_group in enumerate(layer_groups):
        print(f"\n=== Training layer group {group_index}: {layer_group} ===")
        for layer_index in layer_group:
            layer_output_dir = output_dir / f"layer_{layer_index:02d}"
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            train_layer(
                args=args,
                gemma=gemma,
                tokenizer=tokenizer,
                device=device,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                layer_index=layer_index,
                layer_output_dir=layer_output_dir,
            )


def probe_residual(model, tokenizer, sae, text: str, device: str):
    """
    SAE를 모델에 삽입하지 않고 원래 Gemma layer output을 관찰합니다.
    """
    latent_activations = []

    def encode_residual_stream(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output

        with torch.no_grad():
            latent = sae.encode(hidden_states)

        latent_activations.append(latent.detach().cpu())
        return output

    target_layer = get_layer(model, LAYER_INDEX)
    hook = target_layer.register_forward_hook(encode_residual_stream)

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(device)

        model.eval()

        with torch.no_grad():
            model(**inputs)

    finally:
        hook.remove()

    if not latent_activations:
        raise RuntimeError("Probe hook did not collect SAE activations.")

    return latent_activations[0]


def probe(args):
    model, tokenizer, device = load_model_and_tokenizer()
    model.to(device)
    model.eval()

    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=device,
    )
    sae.eval()

    validate_model_sae_compatibility(model, sae)

    latent = probe_residual(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        text=args.text,
        device=device,
    )

    print("SAE hook:", sae.cfg.metadata.hook_name)
    print("HF hook:", sae.cfg.metadata.hf_hook_name)
    print("SAE latent shape:", tuple(latent.shape))
    print("Mean L0:", (latent > 0).float().sum(dim=-1).float().mean().item())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run an SAE activation probe instead of training.",
    )

    parser.add_argument(
        "--text",
        default="Explain why the sky appears blue.",
        help="Text used with --probe.",
    )

    parser.add_argument(
        "--output-dir",
        default="./gemma1b-sae-output",
    )

    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Train one layer; default trains all model layers.",
    )

    parser.add_argument(
        "--layer-group-size",
        type=int,
        default=LAYER_GROUP_SIZE,
        help="Number of layers in each training group.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--recon-loss-weight",
        type=float,
        default=1.0,
        help="Weight for SAE reconstruction MSE loss.",
    )

    parser.add_argument(
        "--sparsity-loss-weight",
        type=float,
        default=1e-5,
        help="Weight for SAE latent L1 sparsity loss.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.probe:
        probe(args)
    else:
        train(args)


if __name__ == "__main__":
    main()