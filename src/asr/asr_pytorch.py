#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import pickle
import numpy as np

# chainer related
import chainer
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

import torch

# spnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import converter_kaldi, converter_mt
from asr_utils import delete_feat, delete_feat_mt
from asr_utils import make_batchset
from asr_utils import restore_snapshot
from e2e_asr_attctc_th import E2E
from e2e_asr_attctc_th import Loss

# for kaldi io
import kaldi_io_py
import lazy_io

# rnnlm
import lm_pytorch

# numpy related
import matplotlib
matplotlib.use('Agg')


class PytorchSeqEvaluaterKaldi(extensions.Evaluator):
    '''Custom evaluater with Kaldi reader for pytorch'''

    def __init__(self, model, iterator, target, reader, device):
        super(PytorchSeqEvaluaterKaldi, self).__init__(
            iterator, target, device=device)
        self.reader = reader
        self.model = model

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                # read scp files
                # x: original json with loaded features
                #    will be converted to chainer variable later
                # batch only has one minibatch utterance, which is specified by batch[0]
                #x = converter_kaldi(batch[0], self.reader)
                x = converter_mt(batch[0])
                self.model.eval()
                self.model(x)
                #delete_feat(x)
                delete_feat_mt(x)

            summary.add(observation)

        return summary.compute_mean()


class PytorchSeqUpdaterKaldi(training.StandardUpdater):
    '''Custom updater with Kaldi reader for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter, optimizer, reader, device):
        super(PytorchSeqUpdaterKaldi, self).__init__(
            train_iter, optimizer, device=None)
        self.model = model
        self.reader = reader
        self.grad_clip_threshold = grad_clip_threshold
        self.num_gpu = len(device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.__next__()

        # read scp files
        # x: original json with loaded features
        #    will be converted to chainer variable later
        # batch only has one minibatch utterance, which is specified by batch[0]
        #x = converter_kaldi(batch[0], self.reader)
        video_ids = [x[0].encode('ascii') for x in batch[0]]
        x = converter_mt(batch[0])

        # Compute the loss at this time step and accumulate it
        loss = self.model(x)
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.num_gpu > 1:
            loss.backward(torch.ones(self.num_gpu))  # Backprop
        else:
            loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        #delete_feat(x)
        delete_feat_mt(x)


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['idim'])
    odim = int(valid_json[utts[0]]['odim'])
    idim = 300
    odim = 50003

    # specify model architecture
    e2e = E2E(idim, odim, args)
    model = Loss(e2e, args.mtlalpha)

    # initialize model by weights of another model
    if args.initchar != 'false':
        logging.info('Initializing weight from model: ' + str(args.initchar))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.initchar)
        for k, v in state_dict.items():
            # do not initialize 'predictor.ctc.ctc_lo.weight', 'predictor.ctc.ctc_lo.bias'
            if 'ctc' in k:
                continue
            # do not initialize 'predictor.dec.embed.weight'
            if 'dec.embed.weight' in k:
                continue
            # do not initialize 'predictor.dec.output.weight', 'predictor.dec.output.bias'
            if 'predictor.dec.output' in k:
                continue
            # do not initialize 'predictor.dec.decoder.0.weight_ih'
            if 'predictor.dec.decoder.0.weight_ih' in k:
                continue
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((idim, odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    reporter = model.reporter
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model = torch.nn.DataParallel(model, device_ids=gpu_id)
        model.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    else:
        gpu_id = [-1]

    # Setup an optimizer
    if args.opt == 'a8dadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_label, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # prepare Kaldi reader
    train_reader = lazy_io.read_dict_mt(args.train_feat, 'tokens_en') #lazy_io.read_dict_scp(args.train_feat)
    valid_reader = lazy_io.read_dict_mt(args.valid_feat, 'tokens_en') #lazy_io.read_dict_scp(args.valid_feat)

    # Set up a trainer
    updater = PytorchSeqUpdaterKaldi(
        model, args.grad_clip, train_iter, optimizer, train_reader, gpu_id)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(PytorchSeqEvaluaterKaldi(
        model, valid_iter, reporter, valid_reader, device=gpu_id))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    def torch_save(path, _):
        torch.save(model.state_dict(), path)
        torch.save(model, path + ".pkl")

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                   trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    logging.warning('Loaded MOdel')

    # epsilon decay in the optimizer
    def torch_load(path, obj):
        model.load_state_dict(torch.load(path))
        return obj
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(100, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(100, 'iteration'))

    trainer.extend(extensions.ProgressBar())

    logging.warning('Start Training')
    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)

    def cpu_loader(storage, location):
        return storage
    model.load_state_dict(torch.load(args.model, map_location=cpu_loader))

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), 650))
        rnnlm.load_state_dict(torch.load(args.rnnlm, map_location=cpu_loader))
    else:
        rnnlm = None

    if args.adaptation == 3:
        idim = idim + 200

    # prepare Kaldi reader
    #reader = kaldi_io_py.read_mat_ark(args.recog_feat)
    reader = lazy_io.read_dict_mt(args.recog_feat, 'tokens_en')
    logging.info("Done reading")

    # read json data
    with open(args.recog_label, 'rb') as f:
        recog_json = json.load(f)['utts']

    #pred_file = open('/data/ASR5/zpp/espnet/egs/howto/exp/pred.txt', 'w')
    #gt_file = open('/data/ASR5/zpp/espnet/egs/howto/exp/gt.txt', 'w')

    new_json = {}
    for name, feat in reader.loader_dict.items():
        if args.beam_size == 1:
            if args.adaptation in [6,7]:
                topic_feat = np.fromstring(recog_json[name]['topic_feat'], dtype=np.float32, sep=' ')
                y_hat = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm, vis_feats=topic_feat)
            elif args.adaptation != 0:
                obj_feat = np.fromstring(recog_json[name]['obj_feat'], dtype=np.float32, sep=' ')
                plc_feat = np.fromstring(recog_json[name]['plc_feat'], dtype=np.float32, sep=' ')
                vis_feat = np.append(obj_feat, plc_feat)
                y_hat = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm, vis_feats=vis_feat)
            else:
                y_hat = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm)

        else:
	    if args.adaptation in [6,7]:
                topic_feat = np.fromstring(recog_json[name]['topic_feat'], dtype=np.float32, sep=' ')
                nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm, vis_feats=topic_feat)
            elif args.adaptation != 0:
                obj_feat = np.fromstring(recog_json[name]['obj_feat'], dtype=np.float32, sep=' ')
                plc_feat = np.fromstring(recog_json[name]['plc_feat'], dtype=np.float32, sep=' ')
                vis_feat = np.append(obj_feat, plc_feat)
                y_hat = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm, vis_feats=vis_feat)
            else:
                nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm)

            # get 1best and remove sos
            y_hat = nbest_hyps[0]['yseq'][1:]

        if name not in recog_json.keys():
            logging.warning('Skipping utt '+name+' as vis feat missing')
            continue

        #y_true = map(int, recog_json[name]['tokenid'].split())
        y_true = map(int, recog_json[name]['tokens_pt'].encode('ascii', 'ignore').split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat if int(idx) < len(train_args.char_list) and train_args.char_list[int(idx)] != '<eos>']
        seq_true = [train_args.char_list[int(idx)] for idx in y_true if int(idx) < len(train_args.char_list) and train_args.char_list[int(idx)] != '<eos>']
        seq_hat_text = " ".join(seq_hat).replace('<space>', ' ')
        seq_true_text = " ".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)
        #pred_file.write(seq_hat_text.encode('ascii', 'ignore') + '\n')
        #gt_file.write(seq_true_text.encode('ascii', 'ignore') + '\n')

        # copy old json info
        new_json[name] = recog_json[name]

        # added recognition results to json
        logging.debug("dump token id")
        # TODO(karita) make consistent to chainer as idx[0] not idx
        new_json[name]['rec_tokenid'] = " ".join([str(idx) for idx in y_hat])
        logging.debug("dump token")
        new_json[name]['rec_token'] = " ".join(seq_hat)
        logging.debug("dump text")
        new_json[name]['rec_text'] = seq_hat_text

        # add n-best recognition results with scores
        if args.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] = " ".join([str(idx) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = hyp['score']

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))


    #pred_file.close()
    #gt_file.close()
