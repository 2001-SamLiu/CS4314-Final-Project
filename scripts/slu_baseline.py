#coding=utf8
import sys, os, time, gc
from xpinyin import Pinyin
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging


# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'augmented_train_with_ontology.json')
dev_path = os.path.join(args.dataroot, 'development.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
test_dataset = Example.load_dataset(test_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

model = SLUTagging(args).to(device)
if args.testing:
    model_info = torch.load('model.bin')
    model.load_state_dict(model_info['model'])
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        predictions = anti_noise_prediction(predictions)
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

def anti_noise_prediction(predictions):
    p = Pinyin()
    select_pos_set = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}
    select_others_set = {'请求类型': [Example.label_vocab.request_map_dic, Example.label_vocab.request_pinyin_set], \
        '出行方式' : [Example.label_vocab.travel_map_dic, Example.label_vocab.travel_pinyin_set], \
        '路线偏好' : [Example.label_vocab.route_map_dic, Example.label_vocab.route_pinyin_set], \
        '对象' :  [Example.label_vocab.object_map_dic, Example.label_vocab.object_pinyin_set], \
        '页码' : [Example.label_vocab.page_map_dic, Example.label_vocab.page_pinyin_set], \
        '操作' : [Example.label_vocab.opera_map_dic, Example.label_vocab.opera_pinyin_set], \
        '序列号' : [Example.label_vocab.ordinal_map_dic, Example.label_vocab.ordinal_pinyin_set]   }

    modify_num = 0
    for i, pred in enumerate(predictions):
        pred_length = len(pred)
        if pred_length > 0 :
            for j in range(pred_length):
                tmp_pred = pred[j]
                split_result = tmp_pred.split('-')
                tmp_pinyin = p.get_pinyin(split_result[2], ' ')
                if split_result[1] != 'value' :
                    if split_result[1] in select_pos_set :
                        map_dic, pinyin_set = Example.label_vocab.poi_map_dic, Example.label_vocab.poi_pinyin_set
                    else :
                        [map_dic, pinyin_set] = select_others_set[split_result[1]]

                    standard_output = get_standard_output (map_dic, pinyin_set, tmp_pinyin)
                    modify_pred = split_result[0] + '-' + split_result[1] + '-' + standard_output
                    if standard_output != split_result[2] :
                        modify_num += 1
                    predictions[i][j] = modify_pred
    print ("modify_num == ", modify_num)                    
    return  predictions            

def get_standard_output (map_dic, pinyin_set, tmp_pinyin) :
    if tmp_pinyin in pinyin_set :
        standard_output = map_dic[tmp_pinyin]
    else :
        max_similarity = 0
        most_similar_pinyin = ''
        for standard_pinyin in iter(pinyin_set) :
            similarity = get_pinyin_similarity(standard_pinyin, tmp_pinyin)
            if similarity > max_similarity :
                max_similarity = similarity
                most_similar_pinyin = standard_pinyin
        if max_similarity == 0 : 
            standard_output = '无'
        else :
            standard_output = map_dic[most_similar_pinyin]
    return standard_output
            

def get_pinyin_similarity(standard_pinyin, tmp_pinyin) :
    standard_set = set (standard_pinyin.split(' '))
    tmp_set = set (tmp_pinyin.split(' '))

    inter_set = standard_set & tmp_set
    similarity = len (inter_set) / (len (standard_set) + len (tmp_set) )
    return similarity


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('test')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
