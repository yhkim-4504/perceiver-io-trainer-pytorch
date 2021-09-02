import train

if __name__ == '__main__':
    # Load
    trainer = train.Trainer()
    trainer.load_dataloader()
    trainer.load_model(state_dict_path=None)
    trainer.load_loss_layer()
    trainer.load_optimizer_scheduler()
    print('\n[Trainer Info]')
    print(trainer)

    # Train
    trainer.train(print_plot=True)  # If you are training in Command Line Interface, set print_plot to False.

    if False:  # Run this when you want to see plots separately.
        print('Testset RMSE Error Plot')
        trainer.plot_dataset(trainer.test_loader)
        trainer.plot_losses()