from T_Eva.SCL import SCLPipeline

if __name__ == "__main__":
    source_file = "../datasets/amazon_d-e/train_all.csv"
    target_file = "../datasets/amazon_d-e/test_all.csv"
    validation_file = "../datasets/amazon_d-e/newtest_all.csv"

    pipeline = SCLPipeline(source_file, target_file, validation_file)

    aux_params = {
        'epochs': 20,
        'batch_size': 4,
        'lr': 0.005
    }

    target_params = {
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
        'hidden_dim': 256,
        'output_dim': 1
    }

    pipeline.run(aux_params=aux_params, target_params=target_params)

