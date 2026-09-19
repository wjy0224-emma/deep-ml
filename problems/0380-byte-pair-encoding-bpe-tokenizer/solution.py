def count_pairs(corpus):
    counts={}

    for word,freq in corpus.items():
        tokens=word.split()

        for i in range(len(tokens)-1):
            pair=(tokens[i],tokens[i+1])

            counts[pair]=counts.get(pair,0)+freq
        
    return counts

def merge_pair(corpus,pair):
    new_corpus={}

    for word,freq in corpus.items():
        tokens=word.split()

        new_tokens=[]
        i=0

        while i<len(tokens):
            if(i<len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]):
                new_tokens.append(pair[0]+pair[1])
                i+=2
            else:
                new_tokens.append(tokens[i])
                i+=1
    
        new_word= ' '.join(new_tokens)
        new_corpus[new_word]=freq

    return new_corpus

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merged_pairs=[]
    for _ in range(num_merges):
        counts=count_pairs(corpus)

        if not counts: break

        best_pair=max(counts,key=counts.get)

        merged_pairs.append(best_pair)

        corpus=merge_pair(corpus,best_pair)

    return merged_pairs


