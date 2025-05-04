from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch
import psutil

class MemoryUsageCallback(TrainerCallback):
    def __init__(self, logger):
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        self.logger = logger

    def log_mem(self,message):
        if torch.cuda.is_available():
            t = torch.cuda.mem_get_info()
            free_gpu, total_gpu = (t[0]/(1024**3),t[1]/(1024**3))
            used_cpu = (psutil.virtual_memory()[3]/1024**3)
            perc_memory = psutil.virtual_memory()[2]/100
            free_cpu_perc = 1 - perc_memory
            total_cpu = (1/perc_memory)*used_cpu
            free_cpu = total_cpu * free_cpu_perc
            self.logger.info(f"*** GPU Memory {message}: {free_gpu:2.0f} GB free from {total_gpu:2.0f} GB  |  "
                        f" CPU Memory: {free_cpu:2.0f} GB free from {total_cpu:2.0f} GB")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            # Optionally reset stats if needed, e.g., if tracking per-step peaks
            # torch.cuda.reset_peak_memory_stats()
            # Get current step's peak
            self.peak_memory_allocated = max(self.peak_memory_allocated, torch.cuda.max_memory_allocated())
            self.peak_memory_reserved = max(self.peak_memory_reserved, torch.cuda.max_memory_reserved())

        # Log if needed (Trainer already logs to W&B/Tensorboard if configured)
            self.log_mem("on step end")
            self.logger.info(f"cuda max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):2.0f} GB")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            self.logger.info("\n--- Training Complete ---")
            self.logger.info(f"Overall Peak VRAM Allocated: {self.peak_memory_allocated / (1024**3):.3f} GB")
            self.logger.info(f"Overall Peak VRAM Reserved: {self.peak_memory_reserved / (1024**3):.3f} GB")
            self.log_mem("on train end")