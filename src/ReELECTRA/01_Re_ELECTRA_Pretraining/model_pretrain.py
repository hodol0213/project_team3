import os
import re
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from transformers import (
    Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl,
    ElectraForMaskedLM, ElectraForPreTraining
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CONFIG

# ---------- utils ----------
def find_last_checkpoint(dirpath: str = CONFIG["OUTPUT_DIR"]):
    if not os.path.exists(dirpath):
        return None
    candidates = [d for d in os.listdir(dirpath) if re.match(r"checkpoint-\d+", d)]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(dirpath, latest)

# ---------- Callbacks ----------
class ElectraLoggingCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        if len(state.log_history) == 0:
            return
        last_log = state.log_history[-1]
        if 'gen_loss' in last_log and 'disc_loss' in last_log:
            current_step = last_log.get('step', state.global_step)
            if args.logging_steps and (current_step % args.logging_steps == 0):
                print(f"Step {current_step} | Generator Loss: {last_log['gen_loss']:.4f} | Discriminator Loss: {last_log['disc_loss']:.4f}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % 1000 == 0 and state.global_step > 0:
            control.should_save = True

class ElectraCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.trainer_ref = None

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        trainer_obj = self.trainer_ref or globals().get("trainer")
        if trainer_obj is None:
            return
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        try:
            torch.save(trainer_obj.generator.state_dict(), os.path.join(output_dir, "generator.pt"))
        except Exception as e:
            print("Generator save failed:", e)
        try:
            torch.save(trainer_obj.discriminator.state_dict(), os.path.join(output_dir, "discriminator.pt"))
        except Exception as e:
            print("Discriminator save failed:", e)

# ---------- Custom Trainer ----------
class ElectraTrainer(Trainer):
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        processing_class,
        warmup_steps=0,
        *args,
        **kwargs
    ):
        super().__init__(model=discriminator, optimizers=(gen_optimizer, None), *args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.processing_class = processing_class
        self.bce_loss_fn = BCEWithLogitsLoss()
        self.warmup_steps = warmup_steps
        self.global_step_counter = 0

    def training_step(self, model, inputs, loss_or_steps=None, **kwargs):
        inputs = self._prepare_inputs(inputs)
        generator = self.generator
        discriminator = self.discriminator
        attention_mask = inputs["attention_mask"]
        real_input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        self.global_step_counter += 1

        # Generator
        self.gen_optimizer.zero_grad()
        gen_outputs = generator(input_ids=real_input_ids, attention_mask=attention_mask, labels=labels)
        gen_loss = gen_outputs.loss
        if torch.isnan(gen_loss) or torch.isinf(gen_loss):
            return torch.tensor(0.0, device=gen_loss.device)
        self.accelerator.backward(gen_loss, retain_graph=True)

        # Fake generation
        with torch.no_grad():
            gen_predictions = gen_outputs.logits.argmax(dim=-1)
            fake_inputs = real_input_ids.clone()
            mask = real_input_ids == self.processing_class.mask_token_id
            fake_inputs[mask] = gen_predictions[mask]

        # Discriminator
        if self.global_step_counter > self.warmup_steps:
            self.disc_optimizer.zero_grad()
            disc_labels = (real_input_ids != fake_inputs).float()
            disc_outputs = discriminator(input_ids=fake_inputs, attention_mask=attention_mask)
            disc_logits = disc_outputs.logits.squeeze(-1)
            disc_loss = self.bce_loss_fn(disc_logits, disc_labels)

            if torch.isnan(disc_loss) or torch.isinf(disc_loss):
                valid_disc_loss = False
                log_disc_loss = torch.tensor(0.0, device=gen_loss.device)
            else:
                valid_disc_loss = True
                log_disc_loss = disc_loss

            if valid_disc_loss:
                self.accelerator.backward(disc_loss)
            disc_loss = log_disc_loss
        else:
            disc_loss = torch.tensor(0.0, device=gen_loss.device)

        # Clip grads
        torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(discriminator.parameters()),
                                       max_norm=self.args.max_grad_norm)

        # Step optimizers
        self.gen_optimizer.step()
        if self.global_step_counter > self.warmup_steps:
            self.disc_optimizer.step()

        # Scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        total_loss = gen_loss + disc_loss
        self.log({
            "gen_loss": float(gen_loss.detach().cpu().item()),
            "disc_loss": float(disc_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item())
        })
        return total_loss.detach()

# ---------- builders & eval ----------
def init_models_and_optimizers(device: str = "cpu", discriminator_lr: float = CONFIG["DISCRIMINATOR_LR"], generator_lr_factor: float = CONFIG["GENERATOR_LR_FACTOR"]):
    generator = ElectraForMaskedLM.from_pretrained("google/electra-small-generator")
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
    generator_lr = discriminator_lr * generator_lr_factor
    gen_optimizer = AdamW(generator.parameters(), lr=generator_lr)
    disc_optimizer = AdamW(discriminator.parameters(), lr=discriminator_lr)
    generator.to(device)
    discriminator.to(device)
    return generator, discriminator, gen_optimizer, disc_optimizer

def get_training_args(output_dir: str = CONFIG["OUTPUT_DIR"], num_train_epochs: int = CONFIG["NUM_EPOCHS_CUDA"], per_device_train_batch_size: int = CONFIG["BATCH_CUDA"], gradient_accumulation_steps: int = CONFIG["GRAD_ACCUM_CUDA"], dataloader_num_workers: int = CONFIG["DATALOADER_WORKERS_CUDA"]):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        logging_steps=CONFIG["LOGGING_STEPS"],
        save_strategy="steps",
        save_steps=CONFIG["SAVE_STEPS"],
        dataloader_num_workers=dataloader_num_workers,
        report_to="none",
        push_to_hub=False,
    )

def build_trainer(generator, discriminator, gen_optimizer, disc_optimizer, tokenizer, tokenized_datasets, data_collator, training_args, warmup_steps: int = CONFIG["WARMUP_STEPS"]):
    log_cb = ElectraLoggingCallback()
    ckpt_cb = ElectraCheckpointCallback()

    trainer = ElectraTrainer(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        processing_class=tokenizer,
        warmup_steps=warmup_steps,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[log_cb, ckpt_cb]
    )

    ckpt_cb.trainer_ref = trainer
    return trainer

def evaluate_models(generator, discriminator, tokenized_datasets, data_collator, device: str = "cuda"):
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=16, shuffle=False, collate_fn=data_collator)
    generator.eval(); discriminator.eval()
    gen_correct = gen_total = disc_correct = disc_total = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="평가 진행 중"):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            gen_outputs = generator(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
            gen_logits = gen_outputs.logits
            mask = inputs["labels"] != -100
            gen_preds = gen_logits.argmax(dim=-1)
            gen_correct += (gen_preds[mask] == inputs["labels"][mask]).sum().item()
            gen_total += mask.sum().item()
            gen_predictions = gen_logits.argmax(dim=-1)
            fake_inputs = inputs["input_ids"].clone()
            fake_inputs[inputs["labels"] != -100] = gen_predictions[inputs["labels"] != -100]
            disc_labels = (inputs["input_ids"] != fake_inputs).float()
            disc_outputs = discriminator(input_ids=fake_inputs, attention_mask=inputs["attention_mask"])
            disc_logits = torch.sigmoid(disc_outputs.logits.squeeze(-1))
            disc_preds = (disc_logits > 0.5).float()
            disc_correct += (disc_preds == disc_labels).sum().item()
            disc_total += disc_labels.numel()
    gen_accuracy = gen_correct / gen_total if gen_total > 0 else 0.0
    disc_accuracy = disc_correct / disc_total if disc_total > 0 else 0.0
    print(f"\n✅ Generator MLM 정확도: {gen_accuracy*100:.2f}%")
    print(f"✅ Discriminator RTD 정확도: {disc_accuracy*100:.2f}%")
    return gen_accuracy, disc_accuracy
