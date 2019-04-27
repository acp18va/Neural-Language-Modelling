# ======================================================================================================================
# Importing
# ======================================================================================================================

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Setting the seed
torch.manual_seed(4)

# ======================================================================================================================
# Declaring variables
# ======================================================================================================================

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10


test_sentence = "The mathematician ran . , " \
                "The mathematician ran to the store . , " \
                "The physicist ran to the store . , " \
                "The philosopher thought about it . , " \
                "The mathematician solved the open problem .".split(" , ")


print("The test sentences are: \n", test_sentence)

# ======================================================================================================================
# Tokenizing and processing the sentences
# ======================================================================================================================

processed_sentence = []
string_processed_sentence = ""
for markers in test_sentence:
    sentence = "START" + " " + markers + " " + "STOP" + " "
    sentences = sentence.split()
    processed_sentence.append(sentences)
    string_processed_sentence = string_processed_sentence + sentence
print("\n")
print("The list of sentences where each sentence is represented "
      "as a list of tokens: ", processed_sentence)
print("\n")
print("Complete sentences are: \n", string_processed_sentence)

print("\n")

# Making the vocabulary
vocab = set(string_processed_sentence.split())
word_to_ix = {word: i for i, word in enumerate(vocab)}

# ======================================================================================================================
# Class NGramLanguageModeler, specifying the dimensionality of the network
# ======================================================================================================================


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        # print("The input-layer dimension: ", embeds.shape)
        out = F.relu(self.linear1(embeds))
        # print("The hidden-layer dimension: ", out.shape)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        # print("The output-layer dimension: ", log_probs.shape)
        return log_probs, self.embeddings


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    total_loss = torch.Tensor([0])
    for sent in processed_sentence:
        # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
        trigrams = [([sent[i], sent[i + 1]], sent[i + 2])
                    for i in range(len(sent) - 2)]
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs, embeddings = model(context_var)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


# ======================================================================================================================
# Running the Sanity check with a Sentence
# ======================================================================================================================

# passing the sentence for sanity check
sanityCheckSent = processed_sentence[1]

# creating trigrams for sanity check sentence
sanityCheckTrigrams = [([sanityCheckSent[i], sanityCheckSent[i + 1]], sanityCheckSent[i + 2])
                    for i in range(len(sanityCheckSent) - 2)]

# printing the trigrams
print(sanityCheckTrigrams)

# switching values to keys in dictionary
reversed_word_to_ix = {val: key for key, val in word_to_ix.items()}
print("\n")

print("Word to ix - ", word_to_ix)
print("Inverted word to ix - ", reversed_word_to_ix)

# counter
ticker = 0

# iterating through the trigrams of sanity check sentence
for previous, current in sanityCheckTrigrams:

    context_idxs = [word_to_ix[w] for w in previous]
    context_var = autograd.Variable(torch.LongTensor(context_idxs))
    model.zero_grad()
    log_probs, embeddings = model(context_var)

    max_log_probs = max(log_probs[0])
    key_value = list(log_probs[0]).index(max_log_probs)
    word_value = reversed_word_to_ix[key_value]
    print(previous, current)
    if current == word_value:
        ticker = ticker + 1
print("Percentage correctly predicted: {0:.2}",format(ticker / len(sanityCheckTrigrams)))


# ======================================================================================================================
# Testing the model
# ======================================================================================================================


input_sent = " The ______ solved the open problem. "
word_choice = ["physicist", "philosopher"]
test_dict = {}
e_dict = {}
for w in word_choice:
    possible_sentence = input_sent.replace("______", w).split()
    test_trigrams = [([possible_sentence[i], possible_sentence[i + 1]], possible_sentence[i + 2])
                       for i in range(len(possible_sentence) - 2)]
    c = 0
    for previous, current in test_trigrams:
        context_idxs = [word_to_ix[word] for word in previous]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        model.zero_grad()
        log_probs, embeddings = model(context_var)
        log_probs_sum = sum(list(log_probs[0]))
        c = c + log_probs_sum
    test_dict[w] = c
    print("yo",test_dict)
    tensor_val = torch.LongTensor([word_to_ix[w]])
    value = embeddings(autograd.Variable(tensor_val))
    e_dict[w] = value

reverse_val = {val: key for key, val in test_dict.items()}
predicted_word = reverse_val[max(reverse_val.keys())]
print("The predicted sentence is: ")
print("The", predicted_word, "solved the open problem.")