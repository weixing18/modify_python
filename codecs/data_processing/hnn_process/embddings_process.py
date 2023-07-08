

'''

从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

from sklearn.manifold import TSNE
import numpy as np
import pickle
from gensim.models import KeyedVectors

# 宏定义
max_text = -1000
long_max_text = -10000

#词向量文件保存成bin文件
def save_word2vec_as_bin(input_path, output_path):
    wv_from_text = KeyedVectors.load_word2vec_format(input_path, binary=False)
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(output_path)
    '''n
    读取用一下代码
    model = KeyedVectors.load(embed_path, mmap='r')
    '''

#构建新的词典 和词向量矩阵
def build_new_word_dict_vec(type_vec_path, type_word_path, final_vec_path, final_word_path):
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())

    # 输出词向量
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']
    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]

    for word in total_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            fail_word.append(word)

    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))



    #判断词向量是否正确
    '''
    couunt = 0
    for i in range(4,len(word_dict)):
        if word_vectors[i].all() == model.wv[word_dict[i]].all():
            continue
        else:
            couunt +=1

    print(couunt)
    '''



    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))

    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)


    print("完成")



#得到词在词典中的位置
def get_index(type, text, word_dict):
    location = []
    if type == 'code':
        location.append(1)
        len_c = len(text)
        if len_c + 1 < 350:
            if len_c == 1 and text[0] == 'max_text':
                location.append(2)
            else:
                location.extend([word_dict.get(word, word_dict['UNK']) for word in text])
                location.append(2)
        else:
            location.extend([word_dict.get(text[i], word_dict['UNK']) for i in range(348)])
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == 'long_max_text':
            location.append(0)
        else:
            location.extend([word_dict.get(word, word_dict['UNK']) for word in text])

    return location


#将训练、测试、验证语料序列化
#查询：25 上下文：100 代码：350
def serialize_corpus(word_dict, input_path, output_path):
    with open(input_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for data in corpus:
        qid = data[0]
        Si_word_list = get_index('text', data[1][0], word_dict)
        Si1_word_list = get_index('text', data[1][1], word_dict)
        tokenized_code = get_index('code', data[2][0], word_dict)
        query_word_list = get_index('text', data[3], word_dict)
        block_length = 4
        label = 0

        if len(Si_word_list) > 100:
            Si_word_list = Si_word_list[:100]
        else:
            Si_word_list.extend([0] * (100 - len(Si_word_list)))

        if len(Si1_word_list) > 100:
            Si1_word_list = Si1_word_list[:100]
        else:
            Si1_word_list.extend([0] * (100 - len(Si1_word_list)))

        if len(tokenized_code) < 350:
            tokenized_code.extend([0] * (350 - len(tokenized_code)))
        else:
            tokenized_code = tokenized_code[:350]

        if len(query_word_list) > 25:
            query_word_list = query_word_list[:25]
        else:
            query_word_list.extend([0] * (25 - len(query_word_list)))

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    with open(output_path, 'wb') as file:
        pickle.dump(total_data, file)

def get_new_dict_append(type_vec_path,previous_dict,previous_vec,append_word_path,final_vec_path,final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件

    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    with open(append_word_path, 'r') as f:
        append_word = eval(f.read())

    # 输出词向量

    print(type(pre_word_vec))
    word_dict =  list(pre_word_dict.keys()) #'#其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word =[]
    print(len(append_word))

    rng = np.random.RandomState(None)
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()

    for word in append_word:
        try:

            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            fail_word.append(word)

    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])



    '''
    #判断词向量是否正确
    print("----------------------------")
    couunt = 0

    import operator
    for i in range(159035,len(word_dict)):
        if operator.eq(word_vectors[i].tolist(), model.wv[word_dict[i]].tolist()) == True:
            continue
        else:
            couunt +=1

    print(couunt)
    '''


    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)


    print("完成")


#-------------------------参数配置----------------------------------
#python 词典 ：1121543 300
if __name__ == '__main__':
    # 将词向量文件保存成bin文件
    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt'
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    save_word2vec_as_bin(ps_path, ps_path_bin)

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'
    save_word2vec_as_bin(sql_path, sql_path_bin)

    # 构建最初基于Staqc的词典和词向量
    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'
    build_new_word_dict_vec(ps_path_bin, python_word_pathpython_word_vec_path, python_word_dict_path)

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'
    build_new_word_dict_vec(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # 最后打标签的语料处理
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabeled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlabeled.txt'

    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    staqc_sql_f = '../hnn_process/ulabel_data/staqc/serialized_sql_staqc_unlabeled_data.pkl'

    with open(sql_final_word_dict_path, 'rb') as f:
        sql_word_dict = pickle.load(f)
    serialize_corpus(sql_word_dict, new_sql_staqc, staqc_sql_f)
    serialize_corpus(sql_word_dict, new_sql_large, large_sql_f)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabeled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlabeled.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'
    get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    staqc_python_f = '../hnn_process/ulabel_data/staqc/serialized_python_staqc_unlabeled_data.pkl'

    with open(python_final_word_dict_path, 'rb') as f:
        python_word_dict = pickle.load(f)
    serialize_corpus(python_word_dict, new_python_staqc, staqc_python_f)
    serialize_corpus(python_word_dict, new_python_large, large_python_f)

    print('序列化完毕')







