import nbformat

def update_notebook():
    nb = nbformat.read('notebooks/RecurrentBitNet_V2.ipynb', as_version=4)

    training_prep_markdown = nbformat.v4.new_markdown_cell("## 5. Training Infrastructure\nSetting up the Differential Optimizer and a dummy dataset to test the progressive recurrence loop locally.")
    
    training_prep_code = nbformat.v4.new_code_cell("""# 5a. Dummy Dataset for Micro Testing
from torch.utils.data import DataLoader, TensorDataset

# Create 100 random batches of tokens
num_batches = 100
B, L = 4, 128
dummy_data = torch.randint(0, config.vocab_size, (num_batches * B, L)).to(DEVICE)
dummy_targets = torch.randint(0, config.vocab_size, (num_batches * B, L)).to(DEVICE)

dataset = TensorDataset(dummy_data, dummy_targets)
dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

# 5b. Differential Optimizer
def create_optimizer(model, base_lr=1e-3):
    # As per DESIGN.md:
    # Encoder/Decoder: standard LR (base_lr)
    # Reasoning Core: higher LR (2 * base_lr)
    # Embeddings: lower LR (0.5 * base_lr)
    
    param_groups = {
        "encoder": [],
        "decoder": [],
        "reasoning": [],
        "embedding": [],
        "other": []
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "encoder" in name:
            param_groups["encoder"].append(param)
        elif "decoder" in name:
            param_groups["decoder"].append(param)
        elif "reasoning_core" in name:
            param_groups["reasoning"].append(param)
        elif "token_emb" in name or "lm_head" in name:
            param_groups["embedding"].append(param)
        else:
            param_groups["other"].append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": param_groups["encoder"], "lr": base_lr},
        {"params": param_groups["decoder"], "lr": base_lr},
        {"params": param_groups["reasoning"], "lr": base_lr * 2.0},
        {"params": param_groups["embedding"], "lr": base_lr * 0.5},
        {"params": param_groups["other"], "lr": base_lr}
    ], weight_decay=0.1)
    
    return optimizer

optimizer = create_optimizer(model, base_lr=1e-3)
print("Optimizer created with differential learning rates.")""")

    training_loop_markdown = nbformat.v4.new_markdown_cell("## 6. Progressive Recurrence Training Loop\nExecuting the training loop that dynamically updates $R$ and calculates the auxiliary loss factor per recurrence pass.")
    
    training_loop_code = nbformat.v4.new_code_cell("""# 6. Progressive Training Loop
from tqdm.auto import tqdm

epochs = 1
alpha = 0.3 # Decay factor for auxiliary loss

# Progressive curriculum schedule: (step_threshold, recurrence_depth)
curriculum = [
    (0, 1),   # Start flat
    (20, 2),  # Shallow recurrence
    (50, 3)   # Deep recurrence
]

print("Starting Micro Training Run...")
global_step = 0
model.train()

loss_history = []
recurrence_history = []

for epoch in range(epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, (idx, targets) in enumerate(pbar):
        # 1. Update curriculum
        for threshold, R in reversed(curriculum):
            if global_step >= threshold:
                model.config.reasoning_recurrence = R
                break
                
        optimizer.zero_grad()
        
        # 2. Forward pass
        logits, base_loss, iter_outputs = model(idx, targets)
        
        # 3. Auxiliary Loss Calculation
        # L_tot = L_final + sum(alpha^(R-r) * L_step_r)
        aux_loss = 0.0
        R = model.config.reasoning_recurrence
        
        # Calculate loss for each intermediate step representation
        for r, hidden in enumerate(iter_outputs):
            # Normalization and tie weight projection
            step_normed = model.final_norm(hidden)
            step_logits = model.lm_head(step_normed)
            step_loss = F.cross_entropy(step_logits.view(-1, step_logits.size(-1)), targets.view(-1))
            
            # Apply decay weight (alpha^(R-r))
            decay_weight = alpha ** (R - (r + 1))
            aux_loss += decay_weight * step_loss
            
        total_loss = base_loss + aux_loss
        
        # 4. Backward Pass
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        global_step += 1
        loss_history.append(total_loss.item())
        recurrence_history.append(R)
        
        pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "R": R})

print("\\nTraining Complete!")""")

    plot_markdown = nbformat.v4.new_markdown_cell("## 7. Results & Visualization")

    plot_code = nbformat.v4.new_code_cell("""import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Loss
ax1.plot(loss_history, label="Total Loss (Base + Aux)", color='blue')
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss Over Steps")
ax1.grid(alpha=0.3)

# Plot Recurrence Curriculum
ax2.step(range(len(recurrence_history)), recurrence_history, where='post', color='red', label="Recurrence Depth (R)")
ax2.set_ylabel("Recurrence (R)")
ax2.set_xlabel("Training Steps")
ax2.set_yticks([1, 2, 3])
ax2.set_title("Progressive Recurrence Depth Curriculum")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()""")

    new_cells = [
        training_prep_markdown,
        training_prep_code,
        training_loop_markdown,
        training_loop_code,
        plot_markdown,
        plot_code
    ]
    nb.cells.extend(new_cells)

    with open('notebooks/RecurrentBitNet_V2.ipynb', 'w') as f:
        nbformat.write(nb, f)
    print("Training cells added.")
    
if __name__ == '__main__':
    update_notebook()
