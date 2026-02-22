import pandas as pd
from src.tokenizer import Tokenizer

def main():
    df = pd.read_csv("./data/data.csv")
    tokenizer = Tokenizer.from_pretrained("english")
    tokens = tokenizer.encode("Hello World")

if __name__ == "__main__":
    main()
