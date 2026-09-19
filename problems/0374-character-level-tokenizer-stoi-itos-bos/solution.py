class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        self.stoi={
            '<BOS>':0,
            '<EOS>':1
        }
        chars=sorted(set(text))

        for i,ch in enumerate(chars,start=2):
            self.stoi[ch]=i 
        
        self.itos={v:k for k,v in self.stoi.items()}

        self.vocab_size=len(self.stoi)


    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        # Your code here
        result=[self.stoi['<BOS>']]

        for s in text:
            result.append(self.stoi[s])

        result.append(self.stoi['<EOS>'])
        return  result

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        # Your code here
        result=''
        for idx in indices:
                result+=self.itos[idx]

        return result
