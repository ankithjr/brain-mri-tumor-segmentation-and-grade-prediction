
import os

from pathlib import Path



root = Path("/scratch/markov3/users/arj94/brain-glioma")

os.environ["nnUNet_raw"] = str(root / "nnUNet_raw")

os.environ["nnUNet_preprocessed"] = str(root / "nnUNet_preprocessed")

os.environ["nnUNet_results"] = str(root / "checkpoints/smoke-test")

os.environ["nnUNet_compile"] = "false"

os.environ["nnUNet_n_proc_DA"] = "2"

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"

os.environ["OPENBLAS_NUM_THREADS"] = "1"



import json

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer



def main():

    torch.set_num_threads(2)

    folder = root / "nnUNet_preprocessed/Dataset501_UCSFPDGM"

    plans = json.loads((folder / "nnUNetPlans.json").read_text())

    dataset = json.loads((folder / "dataset.json").read_text())

    split = json.loads((folder / "splits_final.json").read_text())[0]



    assert len(split["train"]) == 346

    assert len(split["val"]) == 74

    assert not set(split["train"]) & set(split["val"])

    assert torch.cuda.is_available()



    plans["continue_training"] = False
    trainer = nnUNetTrainer(

        plans=plans,

        configuration="3d_fullres",

        fold=0,

        dataset_json=dataset,

        device=torch.device("cuda", 0),

    )

    trainer.num_epochs = 1

    trainer.num_iterations_per_epoch = 5

    trainer.num_val_iterations_per_epoch = 2

    trainer.disable_checkpointing = True



    torch.cuda.reset_peak_memory_stats()

    trainer.run_training()

    torch.cuda.synchronize()



    print("SMOKE TEST COMPLETED", flush=True)

    print(

        "Peak allocated GPU memory (GiB):",

        round(torch.cuda.max_memory_allocated() / 1024**3, 2),

        flush=True,

    )



if __name__ == "__main__":

    main()

