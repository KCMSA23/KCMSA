
def load_twitter():
    sentences = []
    labels = []
    with open("Twitter_dataset/tweets.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("Twitter_dataset/label2.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line == "Neutral":
                labels.append(0)
            elif line == "Positive":
                labels.append(1)
            elif line == "Negative":
                labels.append(2)
    return sentences, labels

def load_twitter_train():
    sentences = []
    labels = []
    with open("Twitter_dataset/tweets_train.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("Twitter_dataset/label2_train.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line == "Neutral":
                labels.append(0)
            elif line == "Positive":
                labels.append(1)
            elif line == "Negative":
                labels.append(2)
    return sentences, labels

def load_twitter_test():
    sentences = []
    labels = []
    with open("Twitter_dataset/tweets_test.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("Twitter_dataset/lable2_test.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line == "Neutral":
                labels.append(0)
            elif line == "Positive":
                labels.append(1)
            elif line == "Negative":
                labels.append(2)
    return sentences, labels

def load_weibo_train():
    sentences = []
    labels = []
    with open("KCMSA_dataset/KCMSA_train_data.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("KCMSA_dataset/KCMSA_train_label.txt", encoding="utf-8") as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            line = line.strip("\n")
            if line == "0":
                labels.append(0)
            elif line == "1":
                labels.append(1)
            elif line == "2":
                labels.append(2)
            else:
                print(i)
            i+=1
    return sentences, labels

def load_weibo_test():
    sentences = []
    labels = []
    with open("KCMSA_dataset/KCMSA_test_data.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("KCMSA_dataset/KCMSA_test_label.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line == "0":
                labels.append(0)
            elif line == "1":
                labels.append(1)
            elif line == "2":
                labels.append(2)
    return sentences, labels


def load_weibo_train_prompt():
    sentences = []
    labels = []
    with open("KCMSA_dataset/KCMSA_train_data.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("KCMSA_dataset/KCMSA_train_label.txt", encoding="utf-8") as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            line = line.strip("\n")
            if line == "0":
                labels.append(17284)
            elif line == "1":
                labels.append(5128)
            elif line == "2":
                labels.append(4533)
            else:
                print(i)
            i+=1
    return sentences, labels

def load_twitter_train_prompt():
    sentences = []
    labels = []
    with open("KCMSA_dataset/tweets_train.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            sentences.append(line)
    with open("KCMSA_dataset/label2_train.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line == "Neutral":
                labels.append(17284)
            elif line == "Positive":
                labels.append(5128)
            elif line == "Negative":
                labels.append(4533)
    return sentences, labels


# def load_twitter_prompt():
#     sentences = []
#     labels = []
#     with open("KCMSA_dataset/tweets.txt", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip("\n")
#             sentences.append(line)
#     with open("KCMSA_dataset/label2.txt", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip("\n")
#             if line == "Neutral":
#                 labels.append(17284)
#             elif line == "Positive":
#                 labels.append(5128)
#             elif line == "Negative":
#                 labels.append(4533)
#     return sentences, labels

def get_label_words(template_id):
    # with open("weibodataset/Positive_word_list.txt", "r") as f:
    # with open("weibodataset/Positive_word_list_one.txt", "r") as f:
    with open("Refine/Positive_FR_" + str(template_id) + "_one.txt", "r") as f:
    # with open("Refine/Positive_FR_" + str(template_id) + ".txt", "r") as f:
        postive_words = []
        lines = f.readlines()
        for line in lines[0:20]:
        # for line in lines:
            postive_words.append(line.strip("\n"))

    # with open("weibodataset/Neutral_word_list.txt", "r") as f:
    # with open("weibodataset/Neutral_word_list_one.txt", "r") as f:
    with open("Refine/Neutral_FR_" + str(template_id) + "_one.txt", "r") as f:
    # with open("Refine/Neutral_FR_" + str(template_id) + ".txt", "r") as f:
        neutral_words = []
        lines = f.readlines()
        for line in lines[0:20]:
        # for line in lines: 
            neutral_words.append(line.strip("\n"))

    # with open("weibodataset/Negative_word_list.txt", "r") as f:
    # with open("weibodataset/Negative_word_list_one.txt", "r") as f:
    with open("Refine/Negative_FR_" + str(template_id) + "_one.txt", "r") as f:
    # with open("Refine/Negative_FR_" + str(template_id) + ".txt", "r") as f:
        negative_words = []
        lines = f.readlines()
        for line in lines[0:20]:
        # for line in lines:
            negative_words.append(line.strip("\n"))
    
    return postive_words,neutral_words,negative_words