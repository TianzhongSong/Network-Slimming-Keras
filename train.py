import keras
from utils.load_cifar import load_data
from keras.preprocessing.image import ImageDataGenerator
from models import resnet, VGGnet
from utils.schedules import onetenth_60_120_160
from utils.channel_pruning import freeze_SR_layer, set_compact_model_weights
import argparse
import os
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir, prefix):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(prefix)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(prefix)))
    plt.close()


def save_history(history, result_dir, prefix):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def training():
    batch_size = 64
    epochs = 200
    fine_tune_epochs = 50
    lr = 0.1

    x_train, y_train, x_test, y_test, nb_classes = load_data(args.data)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=5. / 32,
                                 height_shift_range=5. / 32)
    data_iter = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)

    if args.model == 'resnet':
        model = resnet.resnet(nb_classes,
                              depth=args.depth,
                              wide_factor=args.wide_factor,
                              sparse_factor=args.sparse_factor)
        save_name = 'resnet_{}_{}_{}'.format(args.depth, args.wide_factor, args.data)
    else:
        model = VGGnet.vgg(nb_classes, sparse_factor=args.sparse_factor)
        save_name = 'VGGnet_{}'.format(args.data)

    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(data_iter,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  callbacks=[onetenth_60_120_160(lr)],
                                  validation_data=(x_test, y_test))
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    plot_history(history, './results/', save_name)
    save_history(history, './results/', save_name)
    model.save_weights('./results/{}_weights.h5'.format(save_name))

    freeze_SR_layer(model, args.prune_rate)
    # todo: a little change
    opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.save_weights('./results/{}_{}_weights.h5'.format(save_name, 'fine_tuned'))

    model.fit_generator(data_iter,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=fine_tune_epochs,
                        validation_data=(x_test, y_test))
    # create compact model
    if args.model == 'resnet':
        model = resnet.resnet(nb_classes,
                              depth=args.depth,
                              wide_factor=args.wide_factor,
                              sparse_factor=args.sparse_factor, prune_rate=args.prune_rate)
    else:
        model = VGGnet.vgg(nb_classes, sparse_factor=args.sparse_factor, prune_rate=args.prune_rate)
    compact_model.summary()

    set_compact_model_weights(model, compact_model)
    opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    compact_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    score = compact_model.evaluate(x_test, y_test, verbose=0)
    print('loss: {}'.format(score[0]))
    print('acc: {}'.format(score[1]))
    compact_model.save_weights('./results/{}_{}_weights.h5'.format(save_name, 'channel_pruned'))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str, default='c10', help='Supports c10 (CIFAR-10) and c100 (CIFAR-100)')
    parse.add_argument('--model', type=str, default='vgg')
    parse.add_argument('--depth', type=int, default=40)
    parse.add_argument('--growth-rate', type=int, default=12)
    parse.add_argument('--wide-factor', type=int, default=1)
    parse.add_argument('--sparse-factor', type=int, default=1e-5)
    parse.add_argument('--prune-rate', type=int, default=0.75)
    args = parse.parse_args()

    if args.data not in ['c10', 'c100']:
        raise Exception('args.data must be c10 or c100!')

    training()
