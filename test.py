from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    recon_test_loss = 0.0
    for i, data in enumerate(dataset):
        model.set_input(data)
        if opt.dataset_mode == 'reconstruction':
            recon_test_loss += model.test()
            continue
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    recon_test_loss /= len(dataset)
    if opt.dataset_mode == 'reconstruction':
        writer.print_acc(epoch, recon_test_loss)
        return recon_test_loss
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
