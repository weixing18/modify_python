import pickle

#构建初步词典的具体步骤1
def get_vocab(corpus1, corpus2):
    word_vocab = set()
    for i in range(len(corpus1)):
        for j in range(len(corpus1[i][1][0])):
            word_vocab.add(corpus1[i][1][0][j])
        for j in range(len(corpus1[i][1][1])):
            word_vocab.add(corpus1[i][1][1][j])
        for j in range(len(corpus1[i][2][0])):
            word_vocab.add(corpus1[i][2][0][j])
        for j in range(len(corpus1[i][3])):
            word_vocab.add(corpus1[i][3][j])

    for i in range(len(corpus2)):
        for j in range(len(corpus2[i][1][0])):
            word_vocab.add(corpus2[i][1][0][j])
        for j in range(len(corpus2[i][1][1])):
            word_vocab.add(corpus2[i][1][1][j])
        for j in range(len(corpus2[i][2][0])):
            word_vocab.add(corpus2[i][2][0][j])
        for j in range(len(corpus2[i][3])):
            word_vocab.add(corpus2[i][3][j])
    
    print(len(word_vocab))
    return word_vocab


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

#构建初步词典
def vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = eval(f.read())
    
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    word_vocab = get_vocab(total_data1, total_data2)
    
    with open(save_path, 'w') as f:
        f.write(str(word_vocab))


def final_vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
    
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    word_vocab = get_vocab(total_data1, total_data2)
    final_vocab = word_vocab - total_data1
    
    with open(save_path, 'w') as f:
        f.write(str(final_vocab))




if __name__ == "__main__":
    #====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

    # vocab_prpcessing(python_hnn, python_staqc, python_word_dict)
    # vocab_prpcessing(sql_hnn, sql_staqc, sql_word_dict)






