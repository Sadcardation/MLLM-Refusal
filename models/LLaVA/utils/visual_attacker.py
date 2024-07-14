import torch
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim

import random

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).half().cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).half().cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).half().cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).half().cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, threshold=0.01):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.threshold = threshold

    def __call__(self, val_loss):
        if val_loss < self.threshold:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

class Attacker:
    def __init__(self, args, model, targets, device="cuda:0", is_rtp=False):
        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)
        self.plot_path = self.args.model_position + f'/{self.args.log_dir}/plots/'

    def attack_method(self, inputs, num_iter, lr):
        if self.args.attack_mode == "normal_mean":
            return self.attack_benchmark_normal_mean(inputs, num_iter, lr)
        elif self.args.attack_mode == "normal":
            return self.attack_benchmark_normal_random(inputs, num_iter, lr)
        else:
            assert False, "Invalid attack_mode"

    def attack_benchmark_normal_random(self, inputs, num_iter=2000, lr=1 / 255):
        print(
            f"lr: {lr}, num_iter: {num_iter}, epsilon: {self.args.eps}/255, freq_split: {self.args.freq_split}, split_method: {self.args.freq_mask}"
        )
        plot_path = self.plot_path + f'loss_curve_{inputs.id}.png'
        epsilon = self.args.eps / 255
        original_image_tensor = (
            inputs.image_tensor.clone().detach().requires_grad_(False)
        )
        image_tensor_denormal = denormalize(original_image_tensor.cuda())
        pert_tensor_denormal = torch.zeros_like(
            image_tensor_denormal, requires_grad=True
        )

        injected_ids = inputs.injected_ids
        image_size = inputs.image_size
        input_ids = inputs.question_ids
        criteria = torch.nn.CrossEntropyLoss()

        optimizer = optim.SGD([pert_tensor_denormal], lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_iter,
            eta_min=1e-4,  # Maximum number of iterations.
        )  # Minimum learning rate.
        pbar = tqdm.tqdm(total=num_iter + 1, leave=True)
        for t in range(num_iter + 1):
            index = injected_ids.shape[1] - 1
            current_input_ids = random.choice(input_ids).clone()
            lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()

            current_input_ids = torch.cat(
                [current_input_ids, injected_ids[:, :index]], dim=1
            )
    
                
            output = self.model.forward(
                input_ids=current_input_ids,
                images=normalize(image_tensor_denormal + pert_tensor_denormal),
                # image_sizes=[image_size],
                # output_hidden_states=True,
            )

            group_loss = criteria(
                output.logits[0][-(index + 1) :], injected_ids[0, : index + 1]
            )

            grad = torch.autograd.grad(group_loss, pert_tensor_denormal)[0]

            # Optimizer
            if self.args.optimizer in ["PGD", "PGD-No-Scheduler"]:
                pert_tensor_denormal = pert_tensor_denormal - lr * grad
            elif self.args.optimizer in ["FGSM"]:
                pert_tensor_denormal = pert_tensor_denormal - lr * grad.sign()
            else:
                assert False, "Invalid optimizer"

            # Apply L2 norm constraint
            if self.args.l2norm and pert_tensor_denormal.norm(p=2) > epsilon:
                pert_tensor_denormal = (
                    pert_tensor_denormal / pert_tensor_denormal.norm(p=2) * epsilon
                ).detach()
                pert_tensor_denormal.requires_grad_(True)
            # Apply L_infinity norm constraint
            else:
                pert_tensor_denormal = torch.clamp(
                    pert_tensor_denormal, min=-epsilon, max=epsilon
                ).detach()
                pert_tensor_denormal.requires_grad_(True)

            if self.args.optimizer not in ["PGD-No-Scheduler"]:
                scheduler.step()
            target_loss = group_loss.item()
            self.loss_buffer.append(target_loss)

            msg = f"target_loss: {target_loss}"
            pbar.set_description(msg)
            pbar.update(1)
            
            if t % 5 == 0:
                self.plot_loss(plot_path)

        image_tensor_final = image_tensor_denormal + pert_tensor_denormal
        image_tensor_final = image_tensor_final.detach().cpu()
        image_tensor_final = image_tensor_final.squeeze(0)
        return image_tensor_final

    def attack_benchmark_normal_mean(self, inputs, num_iter=2000, lr=1 / 255):
        print("Applying normal attack method")
        print(
            f"lr: {lr}, num_iter: {num_iter}, epsilon: {self.args.eps}/255, freq_split: {self.args.freq_split}, split_method: {self.args.freq_mask}"
        )
        early_stopping = EarlyStopping(patience=30, verbose=True, threshold=0.001)
        plot_path = self.plot_path + f'loss_curve_{inputs.id}.png'
        epsilon = self.args.eps / 255
        original_image_tensor = (
            inputs.image_tensor.clone().detach().requires_grad_(False)
        )
        image_tensor_denormal = denormalize(original_image_tensor.cuda())
        pert_tensor_denormal = torch.zeros_like(
            image_tensor_denormal, requires_grad=False
        )

        injected_ids = inputs.injected_ids
        image_size = inputs.image_size
        input_ids = inputs.question_ids
        criteria = torch.nn.CrossEntropyLoss()

        optimizer = optim.SGD([pert_tensor_denormal], lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_iter,
            eta_min=1e-4,  # Maximum number of iterations.
        )  # Minimum learning rate.
        pbar = tqdm.tqdm(total=num_iter + 1, leave=True)
        for t in range(num_iter + 1):
            index = injected_ids.shape[1] - 1
            lr = scheduler.get_last_lr()[0]
            batch_pert_tensor_denormal = torch.zeros_like(
                pert_tensor_denormal, requires_grad=False
            )
            group_loss = torch.tensor(0.0).to(self.device)
            
            chosen_input_ids = random.sample(input_ids, self.args.q_batch)
            for current_input_ids in chosen_input_ids:
                optimizer.zero_grad()
                temp_pert_tensor_denormal = torch.zeros_like(
                    pert_tensor_denormal, requires_grad=True
                )
                current_input_ids = torch.cat(
                    [current_input_ids, injected_ids[:, :index]], dim=1
                )
                
                output = self.model.forward(
                    input_ids=current_input_ids,
                    images=normalize(image_tensor_denormal + pert_tensor_denormal + temp_pert_tensor_denormal),
                    # image_sizes=[image_size],
                    # output_hidden_states=True,
                )

                loss = criteria(
                    output.logits[0][-(index + 1) :], injected_ids[0, : index + 1]
                )
                group_loss += loss.detach()
                
                grad = torch.autograd.grad(loss, temp_pert_tensor_denormal)[0]

                # Optimizer
                if self.args.optimizer in ["PGD", "PGD-No-Scheduler"]:
                    temp_pert_tensor_denormal = temp_pert_tensor_denormal - lr * grad
                elif self.args.optimizer in ["FGSM"]:
                    temp_pert_tensor_denormal = temp_pert_tensor_denormal - lr * grad.sign()
                else:
                    assert False, "Invalid optimizer"

                batch_pert_tensor_denormal += temp_pert_tensor_denormal.detach()
            pert_tensor_denormal += batch_pert_tensor_denormal / len(chosen_input_ids)
            group_loss /= len(chosen_input_ids)
            # pert_tensor_denormal += batch_pert_tensor_denormal
            
            # Apply L2 norm constraint
            if self.args.l2norm and pert_tensor_denormal.norm(p=2) > epsilon:
                pert_tensor_denormal = (
                    pert_tensor_denormal / pert_tensor_denormal.norm(p=2) * epsilon
                ).detach()
            # Apply L_infinity norm constraint
            else:
                pert_tensor_denormal = torch.clamp(
                    pert_tensor_denormal, min=-epsilon, max=epsilon
                ).detach()

            if self.args.optimizer not in ["PGD-No-Scheduler"]:
                scheduler.step()
            target_loss = group_loss.item()
            early_stopping(target_loss)
            self.loss_buffer.append(target_loss)
            
            if t % 5 == 0:
                self.plot_loss(plot_path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            msg = f"target_loss: {target_loss}"
            pbar.set_description(msg)
            pbar.update(1)

        image_tensor_final = image_tensor_denormal + pert_tensor_denormal
        image_tensor_final = image_tensor_final.detach().cpu()
        image_tensor_final = image_tensor_final.squeeze(0)
        return image_tensor_final

    def plot_loss(self, plot_path):
        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label="Target Loss")

        # Add in a title and axes labels
        plt.title("Loss Plot")
        plt.xlabel("Iters")
        plt.ylabel("Loss")

        # Display the plot
        plt.legend(loc="best")
        plt.savefig(plot_path)
        plt.clf()