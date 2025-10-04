from src.utils.train_start import Train

def main():
    trainer = Train("src/data/raw/participants/new_train_candles.csv")
    trainer.fit()

if __name__ == '__main__':
    main()

