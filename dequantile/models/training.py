import torch
from matplotlib import pyplot as plt

from dequantile.utils.plotting import plot_training


def get_loader(data_object, valid_loader):
    if valid_loader is not None:
        return data_object, valid_loader, valid_loader
    else:
        return data_object.get_loaders()


def train(model, data_object, n_epochs, learning_rate, device, directory, use_scheduler=True, sv_nm=None,
          decorrelator=False, valid_loader=None):
    directory.mkdir(exist_ok=True, parents=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_scheduler:
        train_loader, _, _ = get_loader(data_object, valid_loader)
        num_steps = len(train_loader) * n_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps,
                                                               last_epoch=-1, eta_min=0)

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader),
        #                                                 epochs=n_epochs, pct_start=0.3, anneal_strategy='cos',
        #                                                 cycle_momentum=True, base_momentum=0.85,
        #                                                 max_momentum=0.95, div_factor=10.0,
        #                                                 final_div_factor=1e2, last_epoch=-1)

    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        t_loss = []
        # Redefine the loaders at the start of every epoch, this will also resample if resampling the mass
        train_loader, valid_loader, _ = get_loader(data_object, valid_loader)
        for step, (data, labels, mass, weights) in enumerate(train_loader):
            model.train()
            # Zero the accumulated gradients
            optimizer.zero_grad()
            # Get the loss
            loss = model.compute_loss(data, labels, mass, weights, device)
            # Calculate the derivatives
            loss.backward()
            # Step the optimizers and schedulers
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            # Store the loss
            t_loss += [loss.item()]

        # Need to set the scaling for every model to be able to load with the right parameters
        if hasattr(model, 'set_scaling'):
            model.set_scaling(train_loader, device)

        # Save the model
        torch.save(model.state_dict(), directory / f'{epoch}')

        # Store the losses across the epoch and start the validation
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(valid_loader))
        for v_step, (v_data, v_label, v_mass, v_weight) in enumerate(valid_loader):
            with torch.no_grad():
                v_loss[v_step] = model.compute_loss(v_data, v_label, v_mass, v_weight, device)
        valid_loss[epoch] = v_loss.mean()

        if decorrelator:
            data_object.update_sampler(0 if data_object.resample else 2)

    if sv_nm is not None:
        # Training and validation losses
        fig = plot_training(train_loss, valid_loss)
        fig.savefig(sv_nm)
        plt.close(fig)

    model.eval()
    return train_loss, valid_loss
